---
name: local-youtube-shorts
description: Generate viral-ranked short clips entirely on the local machine from YouTube URLs, hosted videos, or local media. Use when Codex needs to transcribe long-form video with local Whisper, automatically scale clip count by source duration, find strong hooks and payoff moments, diversify selections across the full timeline, preserve the original landscape composition or crop to 9:16, translate subtitles locally, and export social clips without paid APIs.
---

# Local YouTube Shorts

Create high-retention clips without paid APIs. The first run installs dependencies into the private `.venv`; Whisper and translation models are cached locally.

## Run

Use automatic viral selection by default:

```powershell
& "$HOME\.codex\skills\local-youtube-shorts\scripts\run.ps1" `
  -Source "<YouTube URL or local video>" `
  -OutputDir "<output directory>" `
  -SubtitleLanguage zh
```

Defaults: original composition up to 1080p, clips around 90 seconds (80-100 seconds), `small` Whisper on CPU, automatic clip count, and burned subtitles.

Automatic clip targets: up to 6 minutes = 3; 12 minutes = 5; 20 minutes = 6; 30 minutes = 8; 45 minutes = 10; 60 minutes = 12; longer sources scale to a maximum of 20.

## Selection

Rank candidates on a 100-point virality score using:

- Opening hook and direct address.
- Questions, concrete numbers, and named specifics.
- Conflict, contrast, emotion, and contrarian claims.
- Clear payoff or conclusion near the end.
- Information density, complete phrasing, and proximity to the configured duration midpoint (90 seconds by default).
- Per-second voice energy, emphasizing strongly delivered moments.
- Penalties for filler, repetition, weak openings, and fragments.

Use diversity-aware selection to avoid overlapping clips, repeated topics, and highlights clustered in one part of a long video.

## Workflow

1. Confirm `ffmpeg`, `ffprobe`, `yt-dlp`, and Python are available.
2. Run `scripts/run.ps1` with the source and output directory.
3. Read `summary.json` and report rank, score, time range, hook, reason, title, and MP4 path.
4. Use `ranking.json` to inspect the top 100 candidates and score breakdowns.
5. Keep `transcript.json` for later review or reranking.
6. Never claim clips exist until their MP4 files and summary parse successfully.

## Options

- `-NumClips 0`: automatic count; pass `1-30` to override.
- `-MinDuration <seconds>` and `-MaxDuration <seconds>`: override clip duration bounds; ranking automatically targets their midpoint.
- `-Model tiny|base|small|medium|large-v3`: choose speed versus transcription quality.
- `-Language <code>`: force source recognition language.
- `-SubtitleLanguage <code>`: locally translate subtitles, such as `zh`.
- `-TranscriptFile <path>`: reuse a prior `transcript.json` and skip Whisper for fast reranking.
- `-AspectRatio original|portrait`: preserve source composition or create 9:16 face-aware crops.
- `-NoFaceCrop`: use geometric center for portrait output.
- `-KeepSource`: preserve the downloaded source for manual reruns.

Return fewer clips only when there are not enough non-overlapping candidates above the quality floor. Do not pad with fragments or duplicate topics.

## Failure Handling

- If YouTube blocks a download, retry with current `yt-dlp` and Node.js, then surface the exact error.
- If memory is insufficient, retry with `-Model base` or `-Model tiny`.
- If face detection finds nothing, use a centered crop.
- If local translation has no model for the language pair, surface the unsupported pair and do not claim translated subtitles.
