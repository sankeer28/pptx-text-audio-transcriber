#  Audio Transcriber

Extract text and transcribe audio from PowerPoint presentations, MP4 videos, and MP3 files using Whisper.

## Features

- **Text Extraction**: Extracts all text content from PowerPoint slides
- **Audio Transcription**: Uses faster-whisper to transcribe audio from multiple sources:
  - PowerPoint files (.pptx) - embedded audio recordings
  - Video files (.mp4) - audio track extraction
  - Audio files (.mp3) - direct transcription
- **Checkpoint/Resume Support**: Automatically saves progress during long transcriptions
  - Resume from where you left off if interrupted
  - Checkpoints saved every 10 segments
  - Safe to stop with Ctrl+C anytime
- **Live Progress Tracking**: Real-time progress bar and live checkpoint file
  - View transcription progress in `output/[filename]_checkpoint.json`
  - See completed segments as they're processed
  - Track timestamp and text in real-time
- **GPU and CPU Support**: Automatic device detection with intelligent fallback
- **Multiple Models**: Supports various Whisper model sizes (tiny, base, small, medium, large)
- **Configurable**: Easy-to-modify settings for performance and quality tuning

## Requirements

- Python 3.8 or higher
- ffmpeg (for MP4 video processing)
- CUDA-compatible GPU (optional)

## Installation

### 1. Clone or Download the Project

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

#### Windows:
```bash
# Using chocolatey:
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

#### macOS:
```bash
brew install ffmpeg
```

#### Linux:
```bash
sudo apt install ffmpeg
```

## Usage

### 1. Prepare Your Files

- Place your files in the `presentations` folder:
  - PowerPoint presentations (.pptx)
  - Video files (.mp4)
  - Audio files (.mp3)

### 2. Run the Transcriber

```bash
python main.py
```

## Configuration

Edit the configuration settings at the top of `main.py`:

### Transcription Engine
```python
TRANSCRIPTION_ENGINE = "faster-whisper"  # Options: "standard", "faster-whisper"
```

### Folder Settings
```python
PPTX_FOLDER = "presentations"   # Input folder
OUTPUT_FOLDER = "output"        # Output folder
```

### Whisper Model Settings
```python
WHISPER_MODEL = "small"       # Options: "tiny", "base", "small", "medium", "large"
FORCE_LANGUAGE = "en"         # Force language ("en", "es", "fr", etc.) or None
```

### Performance Settings
```python
FORCE_DEVICE = "cpu"          # Options: None (auto), "cpu", "cuda"
USE_HALF_PRECISION = False    # Enable fp16 for speed boost (GPU only)
```

## Model Size Guide

| Model  | Speed | Quality | Memory | Best For |
|--------|-------|---------|--------|----------|
| tiny   | Fastest | Good | ~1GB | Quick drafts, testing |
| base   | Fast | Better | ~1GB | General use |
| small  | Medium | Good | ~2GB | Recommended - best balance |
| medium | Slow | Very Good | ~5GB | High accuracy needs |
| large  | Slowest | Best | ~10GB | Maximum quality |

