from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from urllib.parse import urlparse


HOOK_TERMS = {
    "why", "how", "secret", "mistake", "truth", "never", "best", "worst",
    "because", "imagine", "important", "finally", "actually", "but",
    "为什么", "如何", "秘密", "错误", "真相", "千万", "最", "因为", "但是",
    "关键", "注意", "结果", "竟然", "没想到", "一定", "不要",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=True, text=True, encoding="utf-8", errors="replace")


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, encoding="utf-8", errors="replace").strip()


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def resolve_source(source: str, work: Path, aspect_ratio: str) -> tuple[Path, bool]:
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve(), False
    if not is_url(source):
        raise FileNotFoundError(f"Source is neither a file nor an HTTP URL: {source}")
    template = str(work / "source.%(ext)s")
    video_format = "bv*[height<=1080]+ba/b[height<=1080]" if aspect_ratio == "original" else "bv*+ba/b"
    command = ["yt-dlp", "--no-playlist", "--merge-output-format", "mp4", "-f", video_format]
    if shutil.which("node"):
        command.extend(["--js-runtimes", "node"])
    command.extend(["-o", template, source])
    run(command)
    matches = sorted(work.glob("source.*"))
    if not matches:
        raise RuntimeError("yt-dlp completed without creating a source file")
    return matches[0].resolve(), True


def probe(video: Path) -> dict:
    raw = capture([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(video),
    ])
    data = json.loads(raw)
    stream = data["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration": float(data["format"]["duration"])}


def transcribe(video: Path, model_name: str, language: str | None) -> tuple[list[Segment], str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    items, info = model.transcribe(
        str(video), language=language or None, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 450},
    )
    segments = [Segment(float(x.start), float(x.end), x.text.strip()) for x in items if x.text.strip()]
    if not segments:
        raise RuntimeError("Whisper did not find any spoken content")
    return segments, info.language


def translate_segments(segments: list[Segment], source: str, target: str) -> list[Segment]:
    if source.split("-")[0] == target.split("-")[0]:
        return segments
    import argostranslate.package
    import argostranslate.translate

    source = source.split("-")[0]
    target = target.split("-")[0]
    installed = argostranslate.translate.get_installed_languages()
    from_lang = next((x for x in installed if x.code == source), None)
    to_lang = next((x for x in installed if x.code == target), None)
    translator = from_lang.get_translation(to_lang) if from_lang and to_lang else None
    if translator is None:
        print(f"Downloading offline translation model {source}->{target}...", flush=True)
        argostranslate.package.update_package_index()
        packages = argostranslate.package.get_available_packages()
        package = next((x for x in packages if x.from_code == source and x.to_code == target), None)
        if package is None:
            raise RuntimeError(f"No Argos Translate model is available for {source}->{target}")
        argostranslate.package.install_from_path(package.download())
        installed = argostranslate.translate.get_installed_languages()
        from_lang = next(x for x in installed if x.code == source)
        to_lang = next(x for x in installed if x.code == target)
        translator = from_lang.get_translation(to_lang)
    translated = []
    for index, item in enumerate(segments, 1):
        text = translator.translate(item.text)
        translated.append(Segment(item.start, item.end, text.strip() or item.text))
        if index % 25 == 0:
            print(f"Translated {index}/{len(segments)} subtitle segments", flush=True)
    return translated


def text_score(text: str, duration: float, segment_count: int) -> float:
    lowered = text.lower()
    hooks = sum(1 for term in HOOK_TERMS if term in lowered)
    questions = text.count("?") + text.count("？")
    exclaims = text.count("!") + text.count("！")
    numbers = len(re.findall(r"\d+", text))
    density = min(len(text) / max(duration, 1), 8.0)
    structure = min(segment_count / 8.0, 2.0)
    return hooks * 4.0 + questions * 2.5 + exclaims * 1.2 + numbers * 1.5 + density * 1.8 + structure


def build_candidates(segments: list[Segment], min_duration: float, max_duration: float) -> list[Candidate]:
    result: list[Candidate] = []
    for i, first in enumerate(segments):
        if i and first.start - segments[i - 1].end < 1.0 and i % 2:
            continue
        chosen: list[Segment] = []
        for item in segments[i:]:
            if item.end - first.start > max_duration:
                break
            chosen.append(item)
            duration = item.end - first.start
            if duration >= min_duration:
                text = " ".join(x.text for x in chosen)
                score = text_score(text, duration, len(chosen))
                score += max(0.0, 2.0 - abs(duration - 45.0) / 10.0)
                result.append(Candidate(first.start, item.end, text, score))
    if not result:
        start, end = segments[0].start, min(segments[-1].end, segments[0].start + max_duration)
        text = " ".join(s.text for s in segments if s.start < end)
        result.append(Candidate(start, end, text, text_score(text, end - start, len(segments))))
    return result


def overlap(a: Candidate, b: Candidate) -> float:
    common = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return common / max(1.0, min(a.end - a.start, b.end - b.start))


def choose(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for item in sorted(candidates, key=lambda x: x.score, reverse=True):
        if all(overlap(item, old) < 0.25 for old in selected):
            selected.append(item)
        if len(selected) == count:
            break
    return selected


def detect_face_center(video: Path, candidate: Candidate, width: int) -> float | None:
    import cv2

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    cap = cv2.VideoCapture(str(video))
    centers: list[float] = []
    for t in [candidate.start + (candidate.end - candidate.start) * x / 10 for x in range(1, 10)]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
        if len(faces):
            x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])
            centers.append(float(x + w / 2))
    cap.release()
    return median(centers) if centers else None


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def write_ass(path: Path, candidate: Candidate, segments: list[Segment], width: int = 1080, height: int = 1920) -> None:
    landscape = width > height
    font_size = 48 if landscape else 58
    margin_v = 65 if landscape else 190
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,100,100,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for item in segments:
        if item.end <= candidate.start or item.start >= candidate.end:
            continue
        start = max(item.start, candidate.start) - candidate.start
        end = min(item.end, candidate.end) - candidate.start
        text = item.text.strip()
        if len(text) > 30:
            midpoint = len(text) // 2
            split = max(text.rfind(" ", 0, midpoint), text.rfind("，", 0, midpoint), text.rfind(",", 0, midpoint))
            if split > 8:
                text = text[:split] + "\n" + text[split + 1:]
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{ass_escape(text)}\n")
    path.write_text("".join(lines), encoding="utf-8-sig")


def crop_filter(width: int, height: int, face_x: float | None) -> str:
    target_ratio = 9 / 16
    if width / height >= target_ratio:
        crop_h = height
        crop_w = int(height * target_ratio) // 2 * 2
        center = face_x if face_x is not None else width / 2
        x = int(max(0, min(width - crop_w, center - crop_w / 2)))
        y = 0
    else:
        crop_w = width
        crop_h = int(width / target_ratio) // 2 * 2
        x = 0
        y = max(0, (height - crop_h) // 2)
    return f"crop={crop_w}:{crop_h}:{x}:{y},scale=1080:1920"


def clean_title(text: str, rank: int) -> str:
    title = re.split(r"[。！？.!?]", text.strip(), maxsplit=1)[0].strip()
    title = re.sub(r"\s+", " ", title)
    return (title[:55] or f"Highlight {rank}").strip()


def render(video: Path, out_dir: Path, candidate: Candidate, segments: list[Segment], info: dict, rank: int, face_crop: bool, aspect_ratio: str) -> Path:
    clip_dir = out_dir / f"clip-{rank:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    ass = clip_dir / "subtitles.ass"
    output = clip_dir / f"short-{rank:02d}.mp4"
    if aspect_ratio == "original":
        write_ass(ass, candidate, segments, info["width"], info["height"])
        filters = "subtitles=subtitles.ass"
    else:
        write_ass(ass, candidate, segments)
        face_x = detect_face_center(video, candidate, info["width"]) if face_crop else None
        filters = crop_filter(info["width"], info["height"], face_x) + ",subtitles=subtitles.ass"
    run([
        "ffmpeg", "-y", "-ss", f"{candidate.start:.3f}", "-i", str(video),
        "-t", f"{candidate.end - candidate.start:.3f}", "-vf", filters,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output),
    ], cwd=clip_dir)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local vertical highlight clips")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=30)
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language")
    parser.add_argument("--subtitle-language")
    parser.add_argument("--aspect-ratio", choices=["portrait", "original"], default="portrait")
    parser.add_argument("--no-face-crop", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_duration > args.max_duration:
        raise ValueError("min-duration cannot exceed max-duration")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".work"
    work.mkdir(exist_ok=True)
    video, downloaded = resolve_source(args.source, work, args.aspect_ratio)
    info = probe(video)
    segments, language = transcribe(video, args.model, args.language)
    subtitle_language = args.subtitle_language or language
    segments = translate_segments(segments, language, subtitle_language)
    selected = choose(build_candidates(segments, args.min_duration, args.max_duration), args.num_clips)
    results = []
    for rank, candidate in enumerate(selected, 1):
        output = render(video, out_dir, candidate, segments, info, rank, not args.no_face_crop, args.aspect_ratio)
        results.append({
            "rank": rank,
            "score": round(candidate.score, 2),
            "start_time": round(candidate.start, 3),
            "end_time": round(candidate.end, 3),
            "title": clean_title(candidate.text, rank),
            "transcript_excerpt": candidate.text[:240],
            "path": str(output),
        })
    summary = {
        "source": args.source,
        "language": language,
        "subtitle_language": subtitle_language,
        "model": args.model,
        "aspect_ratio": args.aspect_ratio,
        "video_duration": info["duration"],
        "clips": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if downloaded and not args.keep_source:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
