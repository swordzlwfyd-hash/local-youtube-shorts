from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from urllib.parse import urlparse


HOOK_TERMS = {
    "why", "how", "secret", "mistake", "truth", "never", "best", "biggest", "worst", "nobody", "one thing",
    "because", "imagine", "important", "finally", "actually", "but",
    "为什么", "如何", "秘密", "错误", "真相", "千万", "最", "因为", "但是",
    "关键", "注意", "结果", "竟然", "没想到", "一定", "不要",
}

CONTRAST_TERMS = {
    "but", "however", "instead", "actually", "although", "yet", "wrong",
    "problem", "risk", "fail", "failure", "versus", "difference",
    "但是", "然而", "其实", "反而", "问题", "风险", "失败", "错误", "区别", "真相",
}
PAYOFF_TERMS = {
    "therefore", "so", "result", "lesson", "means", "conclusion", "finally",
    "the point", "what matters", "here's why", "in other words",
    "所以", "因此", "结果", "结论", "关键", "重点", "这意味着", "换句话说", "最终",
}
DIRECT_TERMS = {
    "you", "your", "listen", "remember", "imagine", "here's", "let me",
    "你", "你的", "记住", "想象", "注意", "告诉你", "别", "不要",
}
FILLER_TERMS = {
    "um", "uh", "you know", "kind of", "sort of", "basically", "like",
    "嗯", "呃", "那个", "就是说", "怎么说", "基本上",
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
    hook: str = ""
    reason: str = ""
    breakdown: dict[str, float] = field(default_factory=dict)


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
    command = [
        "yt-dlp", "--no-playlist", "--retries", "10", "--fragment-retries", "10",
        "--merge-output-format", "mp4", "-f", video_format,
    ]
    if shutil.which("node"):
        command.extend(["--js-runtimes", "node", "--remote-components", "ejs:github"])
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


def load_transcript(path: Path) -> tuple[list[Segment], str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(float(item["start"]), float(item["end"]), str(item["text"]).strip())
        for item in data.get("segments", []) if str(item.get("text", "")).strip()
    ]
    if not segments:
        raise ValueError(f"Transcript has no usable segments: {path}")
    language = str(data.get("language") or "auto")
    subtitle_language = str(data.get("subtitle_language") or language)
    return segments, language, subtitle_language


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
    # Whisper already provides short timestamped segments. Bypass Argos sentence
    # detection so translation remains fully offline and preserves those timings.
    if hasattr(translator, "sentencizer"):
        translator.sentencizer.split_sentences = lambda text: [text]
    translated = []
    for index, item in enumerate(segments, 1):
        text = translator.translate(item.text)
        text = re.sub(r"\{\\[^}]*\}", "", text)
        text = re.sub(r"\s*\{\\.*$", "", text)
        translated.append(Segment(item.start, item.end, text.strip() or item.text))
        if index % 25 == 0:
            print(f"Translated {index}/{len(segments)} subtitle segments", flush=True)
    return translated


def count_terms(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    count = 0
    for term in terms:
        lowered_term = term.lower()
        if re.search(r"[a-z]", lowered_term):
            pattern = rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])"
            count += int(bool(re.search(pattern, lowered)))
        else:
            count += int(lowered_term in lowered)
    return count


def first_sentence(text: str, limit: int = 110) -> str:
    sentence = re.split(r"(?<=[.!?。！？])\s*", text.strip(), maxsplit=1)[0]
    return re.sub(r"\s+", " ", sentence)[:limit].strip()


def token_set(text: str) -> set[str]:
    latin = re.findall(r"[a-z0-9]{2,}", text.lower())
    han = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    han_bigrams = [word[i:i + 2] for word in han for i in range(len(word) - 1)]
    return set(latin + han_bigrams)


def score_candidate(text: str, duration: float, segment_count: int, target_duration: float) -> Candidate:
    cleaned = re.sub(r"\s+", " ", text).strip()
    opening = first_sentence(cleaned, 140)
    closing = cleaned[-180:]
    questions = opening.count("?") + opening.count("？")
    exclaims = cleaned.count("!") + cleaned.count("！")
    hook_hits = count_terms(opening, HOOK_TERMS)
    buried_hook_hits = max(0, count_terms(cleaned, HOOK_TERMS) - hook_hits)
    direct_hits = count_terms(opening, DIRECT_TERMS)
    contrast_hits = count_terms(cleaned, CONTRAST_TERMS)
    payoff_hits = count_terms(closing, PAYOFF_TERMS)
    filler_hits = count_terms(cleaned, FILLER_TERMS)
    numbers = len(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?", cleaned))

    hook_score = min(28.0, questions * 7.0 + hook_hits * 4.0 + direct_hits * 3.0 + min(numbers, 2) * 2.5)
    contrast_score = min(15.0, contrast_hits * 3.5)
    specificity_score = min(14.0, numbers * 2.5 + len(re.findall(r"\b[A-Z]{2,}[A-Z0-9-]*\b", text)) * 1.5)
    payoff_score = min(13.0, payoff_hits * 4.0)
    emotion_score = min(9.0, exclaims * 2.0 + questions * 1.5 + count_terms(cleaned, {"love", "hate", "fear", "amazing", "shocking", "喜欢", "害怕", "惊人", "震惊"}) * 2.5)

    char_density = len(re.sub(r"\s", "", cleaned)) / max(duration, 1.0)
    density_score = max(0.0, 10.0 - abs(char_density - 5.0) * 1.4)
    structure_score = min(8.0, segment_count / 2.5)
    if re.search(r"[.!?。！？][\"'”’）)]?$", cleaned):
        structure_score += 3.0
    if re.match(r"^(and|but|so|because|然后|但是|所以|因为)\b", cleaned.lower()):
        structure_score -= 4.0

    duration_tolerance = max(5.0, target_duration * 0.12)
    duration_score = max(0.0, 6.0 - abs(duration - target_duration) / duration_tolerance)
    repeated = re.findall(r"\b([a-z]{2,})\b", cleaned.lower())
    repetition_ratio = 1.0 - len(set(repeated)) / max(1, len(repeated)) if repeated else 0.0
    penalty = min(20.0, filler_hits * 2.5 + max(0.0, repetition_ratio - 0.55) * 20.0)
    penalty += min(10.0, buried_hook_hits * 3.0)

    breakdown = {
        "hook": round(hook_score, 2),
        "contrast": round(contrast_score, 2),
        "specificity": round(specificity_score, 2),
        "payoff": round(payoff_score, 2),
        "emotion": round(emotion_score, 2),
        "density": round(density_score, 2),
        "structure": round(max(0.0, structure_score), 2),
        "duration": round(duration_score, 2),
        "penalty": round(penalty, 2),
    }
    score = max(0.0, min(100.0, sum(value for key, value in breakdown.items() if key != "penalty") - penalty))
    labels = {
        "hook": "开场钩子强", "contrast": "有冲突或反差", "specificity": "包含具体数字或事实",
        "payoff": "结尾有明确回报", "emotion": "情绪张力明显", "density": "信息密度高",
        "structure": "表达完整", "duration": "时长适合短视频",
    }
    strongest = sorted((key for key in labels), key=lambda key: breakdown[key], reverse=True)[:3]
    reason = "、".join(labels[key] for key in strongest if breakdown[key] > 0)
    return Candidate(0.0, duration, cleaned, round(score, 2), first_sentence(cleaned), reason, breakdown)


def natural_start(segments: list[Segment], index: int) -> bool:
    if index == 0:
        return True
    previous = segments[index - 1]
    gap = segments[index].start - previous.end
    return gap >= 0.35 or bool(re.search(r"[.!?。！？][\"'”’）)]?$", previous.text.strip()))


def build_candidates(segments: list[Segment], min_duration: float, max_duration: float) -> list[Candidate]:
    result: list[Candidate] = []
    target_duration = (min_duration + max_duration) / 2.0
    for i, first in enumerate(segments):
        if not natural_start(segments, i):
            continue
        possible: list[Candidate] = []
        chosen: list[Segment] = []
        for item in segments[i:]:
            duration = item.end - first.start
            if duration > max_duration:
                break
            chosen.append(item)
            if duration < min_duration:
                continue
            if duration < max_duration - 4 and not re.search(r"[.!?。！？][\"'”’）)]?$", item.text.strip()):
                continue
            text = " ".join(x.text for x in chosen)
            scored = score_candidate(text, duration, len(chosen), target_duration)
            scored.start = first.start
            scored.end = item.end
            possible.append(scored)
        result.extend(sorted(possible, key=lambda item: item.score, reverse=True)[:2])
    if not result:
        start, end = segments[0].start, min(segments[-1].end, segments[0].start + max_duration)
        text = " ".join(segment.text for segment in segments if segment.start < end)
        candidate = score_candidate(text, end - start, len(segments), target_duration)
        candidate.start, candidate.end = start, end
        result.append(candidate)
    return result


def analyze_audio_energy(video: Path) -> list[float]:
    try:
        import numpy as np

        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
        ])
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        frame_count = len(samples) // 8000
        if frame_count < 2:
            return []
        frames = samples[:frame_count * 8000].reshape(frame_count, 8000)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        logged = np.log1p(rms)
        low, high = np.percentile(logged, [20, 95])
        normalized = np.clip((logged - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        return normalized.tolist()
    except Exception as exc:
        print(f"Audio energy analysis skipped: {exc}", file=sys.stderr, flush=True)
        return []


def apply_audio_scores(candidates: list[Candidate], energy: list[float]) -> None:
    if not energy:
        return
    for item in candidates:
        start = max(0, min(len(energy) - 1, int(item.start)))
        end = max(start + 1, min(len(energy), int(math.ceil(item.end))))
        window = energy[start:end]
        if not window:
            continue
        mean_energy = sum(window) / len(window)
        peak_energy = max(window)
        delivery = round((mean_energy * 0.7 + peak_energy * 0.3) * 10.0, 2)
        item.breakdown["delivery"] = delivery
        item.score = round(min(100.0, item.score + delivery), 2)
        if delivery >= 7.5 and "语音能量突出" not in item.reason:
            item.reason = (item.reason + "、语音能量突出").strip("、")


def overlap(a: Candidate, b: Candidate) -> float:
    common = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return common / max(1.0, min(a.end - a.start, b.end - b.start))


def text_similarity(a: Candidate, b: Candidate) -> float:
    left, right = token_set(a.text), token_set(b.text)
    return len(left & right) / max(1, len(left | right))


def auto_clip_count(duration: float) -> int:
    minutes = duration / 60.0
    if minutes <= 6:
        return 3
    if minutes <= 12:
        return 5
    if minutes <= 20:
        return 6
    if minutes <= 30:
        return 8
    if minutes <= 45:
        return 10
    if minutes <= 60:
        return 12
    return min(20, 12 + math.ceil((minutes - 60) / 15) * 2)


def choose(candidates: list[Candidate], count: int, video_duration: float) -> list[Candidate]:
    remaining = [item for item in candidates if item.score >= 15.0]
    selected: list[Candidate] = []
    spacing = max(75.0, video_duration / max(1, count * 2.2))
    while remaining and len(selected) < count:
        best: Candidate | None = None
        best_adjusted = float("-inf")
        for item in remaining:
            if any(overlap(item, old) >= 0.22 for old in selected):
                continue
            midpoint = (item.start + item.end) / 2
            diversity_penalty = 0.0
            for old in selected:
                old_midpoint = (old.start + old.end) / 2
                proximity = max(0.0, 1.0 - abs(midpoint - old_midpoint) / spacing)
                diversity_penalty = max(diversity_penalty, proximity * 14.0 + text_similarity(item, old) * 22.0)
            adjusted = item.score - diversity_penalty
            if adjusted > best_adjusted:
                best, best_adjusted = item, adjusted
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
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
    parser = argparse.ArgumentParser(description="Generate locally ranked social highlight clips")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-clips", type=int, default=0, help="0 chooses a count from the source duration")
    parser.add_argument("--min-duration", type=float, default=80)
    parser.add_argument("--max-duration", type=float, default=100)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language")
    parser.add_argument("--subtitle-language")
    parser.add_argument("--transcript-file", help="Reuse a transcript.json file and skip Whisper")
    parser.add_argument("--aspect-ratio", choices=["portrait", "original"], default="original")
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
    if args.transcript_file:
        segments, language, stored_subtitle_language = load_transcript(Path(args.transcript_file).expanduser().resolve())
        subtitle_language = args.subtitle_language or stored_subtitle_language
        if subtitle_language != stored_subtitle_language:
            segments = translate_segments(segments, stored_subtitle_language, subtitle_language)
    else:
        segments, language = transcribe(video, args.model, args.language)
        subtitle_language = args.subtitle_language or language
        segments = translate_segments(segments, language, subtitle_language)
    candidates = build_candidates(segments, args.min_duration, args.max_duration)
    audio_energy = analyze_audio_energy(video)
    apply_audio_scores(candidates, audio_energy)
    requested_count = args.num_clips or auto_clip_count(info["duration"])
    print(f"Ranking {len(candidates)} candidates; target clip count: {requested_count}", flush=True)
    selected = choose(candidates, requested_count, info["duration"])
    transcript_path = out_dir / "transcript.json"
    transcript_path.write_text(json.dumps({
        "language": language,
        "subtitle_language": subtitle_language,
        "segments": [{"start": item.start, "end": item.end, "text": item.text} for item in segments],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    ranking_path = out_dir / "ranking.json"
    ranking_path.write_text(json.dumps({
        "candidate_count": len(candidates),
        "audio_scoring": bool(audio_energy),
        "top_candidates": [{
            "start_time": round(item.start, 3),
            "end_time": round(item.end, 3),
            "score": item.score,
            "hook_sentence": item.hook,
            "virality_reason": item.reason,
            "score_breakdown": item.breakdown,
        } for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:100]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for rank, candidate in enumerate(selected, 1):
        output = render(video, out_dir, candidate, segments, info, rank, not args.no_face_crop, args.aspect_ratio)
        results.append({
            "rank": rank,
            "score": round(candidate.score, 2),
            "start_time": round(candidate.start, 3),
            "end_time": round(candidate.end, 3),
            "title": clean_title(candidate.text, rank),
            "hook_sentence": candidate.hook,
            "virality_reason": candidate.reason,
            "score_breakdown": candidate.breakdown,
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
        "selection_mode": "auto" if args.num_clips == 0 else "fixed",
        "requested_clip_count": requested_count,
        "candidate_count": len(candidates),
        "audio_scoring": bool(audio_energy),
        "selected_clip_count": len(results),
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
