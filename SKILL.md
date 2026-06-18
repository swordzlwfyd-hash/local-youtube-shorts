---
name: local-youtube-shorts
description: Generate ranked vertical short videos entirely on the local machine from YouTube URLs, hosted videos, or local media files. Use when Codex needs to download or ingest a long video, transcribe it with local Whisper, select highlight segments, crop them to 9:16 around detected faces, burn Chinese or multilingual subtitles, and export YouTube Shorts, TikTok, or Reels without paid APIs.
---

# Local YouTube Shorts

Create short-form clips without paid APIs. The first run installs local Python dependencies into this skill's private `.venv`; Whisper models are cached locally after download.

## Run

Use PowerShell:

```powershell
& "$HOME\.codex\skills\local-youtube-shorts\scripts\run.ps1" `
  -Source "<YouTube URL or local video>" `
  -OutputDir "<output directory>" `
  -NumClips 3
```

Defaults: `small` Whisper model, CPU `int8`, 30-60 second clips, 3 outputs, `9:16`, burned subtitles. Pass `-Language zh` to force Chinese recognition or omit it for automatic detection. Pass `-SubtitleLanguage zh` to translate foreign speech into Chinese with a locally cached Argos Translate model.

For a quicker CPU run, pass `-Model base`. For better transcription, pass `-Model medium`; warn that it is slower and downloads a larger model.

## Workflow

1. Confirm `ffmpeg`, `ffprobe`, `yt-dlp`, and Python are available.
2. Run `scripts/run.ps1` with the user's source and requested output directory.
3. Let the script finish; the first model download can take several minutes.
4. Read `summary.json` from the output directory.
5. Report each clip's rank, score, time range, title, transcript excerpt, and local MP4 path.
6. Never claim clips exist until their MP4 files and `summary.json` are present.

## Options

- `-NumClips <n>`: requested clip count.
- `-MinDuration <seconds>` and `-MaxDuration <seconds>`: candidate duration bounds.
- `-Model tiny|base|small|medium|large-v3`: local Whisper model.
- `-Language <code>`: force a language, such as `zh`, `en`, or `ja`.
- `-SubtitleLanguage <code>`: locally translate subtitles to a target language such as `zh`; the free translation model downloads once and is then cached.
- `-NoFaceCrop`: use geometric center instead of local face detection.
- `-KeepSource`: preserve the downloaded/intermediate source video.

If fewer non-overlapping high-quality candidates exist, return fewer clips rather than padding the result.

## Failure Handling

- If YouTube blocks a download, surface the `yt-dlp` error and suggest a local file.
- If memory is insufficient, retry with `-Model base` or `-Model tiny`.
- If face detection finds nothing, the pipeline automatically uses a centered crop.
- If subtitle rendering fails because of a missing font, the ASS renderer falls back to an installed sans-serif font.
