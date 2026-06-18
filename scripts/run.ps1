param(
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$OutputDir = (Join-Path (Get-Location) "shorts-output"),
    [ValidateRange(0, 30)][int]$NumClips = 0,
    [ValidateRange(10, 180)][int]$MinDuration = 25,
    [ValidateRange(10, 180)][int]$MaxDuration = 55,
    [ValidateSet("tiny", "base", "small", "medium", "large-v3")][string]$Model = "small",
    [string]$Language = "",
    [string]$SubtitleLanguage = "",
    [string]$TranscriptFile = "",
    [ValidateSet("portrait", "original")][string]$AspectRatio = "original",
    [switch]$NoFaceCrop,
    [switch]$KeepSource
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SkillDir = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $SkillDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

foreach ($tool in @("ffmpeg", "ffprobe", "yt-dlp", "python")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $tool"
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating private Python environment..."
    python -m venv $VenvDir
}

$DependenciesReady = $false
if (Test-Path $Python) {
    & $Python -c "import faster_whisper, cv2" 2>$null
    $DependenciesReady = ($LASTEXITCODE -eq 0)
}
if (-not $DependenciesReady) {
    Write-Host "Installing local transcription and face-detection dependencies..."
    & $Python -m pip install --upgrade pip
    & $Python -m pip install "faster-whisper>=1.1,<2" "opencv-python-headless>=4.10,<5"
}
if ($SubtitleLanguage) {
    & $Python -c "import argostranslate" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing local subtitle translation dependencies..."
        & $Python -m pip install "argostranslate>=1.9,<2"
    }
}

$argsList = @(
    (Join-Path $ScriptDir "pipeline.py"),
    "--source", $Source,
    "--output-dir", $OutputDir,
    "--num-clips", $NumClips,
    "--min-duration", $MinDuration,
    "--max-duration", $MaxDuration,
    "--model", $Model
    "--aspect-ratio", $AspectRatio
)
if ($Language) { $argsList += @("--language", $Language) }
if ($SubtitleLanguage) { $argsList += @("--subtitle-language", $SubtitleLanguage) }
if ($TranscriptFile) { $argsList += @("--transcript-file", $TranscriptFile) }
if ($NoFaceCrop) { $argsList += "--no-face-crop" }
if ($KeepSource) { $argsList += "--keep-source" }

& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "Shorts pipeline failed with exit code $LASTEXITCODE" }
