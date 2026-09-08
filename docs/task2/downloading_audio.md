# Downloading real MusicCaps audio on Windows

The error `ffmpeg is not installed` means yt-dlp cannot extract the requested
MusicCaps interval. The JavaScript runtime warning also needs attention for
reliable YouTube extraction. Install both tools in PowerShell:

```powershell
winget install --exact --id Gyan.FFmpeg --source winget
winget install --exact --id DenoLand.Deno --source winget
```

Restart VS Code after installation so new terminals receive the updated PATH.
From the project root, install yt-dlp into the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "yt-dlp[default]"
```

The [download script](../../scripts/download_musiccaps.ps1) finds FFmpeg,
ffprobe and Deno through the current and persistent Windows PATH values or your
per-user WinGet Links folder and supplies
their locations explicitly to yt-dlp. This also handles many already-open
terminals with an outdated PATH. It can use an existing Node.js version 22 or
newer when Deno 2 or newer is unavailable. Missing dependencies stop the script
before any downloads.

First request one clip:

```powershell
.\scripts\download_musiccaps.ps1 -Limit 1
```

Once that succeeds, download the full metadata list:

```powershell
.\scripts\download_musiccaps.ps1
```

`-Limit` selects the first N metadata rows, including rows whose clips already
exist. Use `-MetadataCsv` and `-AudioDir` to override the defaults. Relative paths
are resolved from the repository root. The defaults are
`data/raw/musiccaps_official.csv` and `data/raw/musiccaps_audio/`.

Each output is named `{ytid}_{start_s}_{end_s}.wav` and contains the requested
interval. The script verifies a WAV audio stream, duration within 0.02 seconds
of the requested interval, and successful decoding of the entire file. It
checks existing WAVs the same way before skipping them, so rerunning resumes
without relying on a video-level download archive. Existing invalid WAVs are
preserved; inspect and move or rename them before retrying.

Failures are logged to `results/task2/audio_download_failures.txt`. Three
consecutive failures stop the download. If the errors show a bot check, rate
limit or access restriction, stop and inspect the error before rerunning.
Deleted or unavailable videos can remain missing; Task 2's manifest reports
which samples are retained. Do not replace missing audio with synthetic clips.

After acquisition, follow the [real-data workflow](README.md#real-data-workflow)
with `--audio-mode pretrimmed`. The one-clip check only verifies acquisition;
training needs available clips in all three dataset splits.

References: [MusicCaps acquisition instructions](https://huggingface.co/datasets/google/MusicCaps#dataset-usage),
[yt-dlp download options](https://github.com/yt-dlp/yt-dlp#download-options),
[yt-dlp JavaScript runtime documentation](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
