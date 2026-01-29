# PPTX Text and Audio Transcriber

Extract text and transcribe audio from PowerPoint presentations, MP4 videos, and MP3 files using OpenAI Whisper.

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

```bash
git clone https://github.com/sankeer28/pptx-text-audio-transcriber.git
cd pptx-text-audio-transcriber
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

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

- Create a `presentations` folder in the project directory
- Place your files in the `presentations` folder:
  - PowerPoint presentations (.pptx)
  - Video files (.mp4)
  - Audio files (.mp3)

### 2. Run the Transcriber

```bash
python main.py
```

### 3. Check Results

- Extracted content will be saved in the `output` folder
- Each file generates a corresponding `.txt` file with transcribed text
- For PowerPoint files, both slide text and audio transcriptions are included

### 4. Resume from Checkpoint (If Interrupted)

If transcription is interrupted:
- Simply run `python main.py` again
- The script will automatically detect and resume from the last checkpoint
- Progress is saved every 10 segments during transcription

### 5. Monitor Live Progress

While transcription is running:
- Open `output/[filename]_checkpoint.json` to see real-time progress
- View completed segments with timestamps and text
- Track current position in the audio

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

**Recommendation**: Use `small` model for best accuracy/speed balance on CPU.

## Checkpoint System

The checkpoint system automatically saves your progress during long transcriptions:

### How It Works
- Progress saved every 10 segments to `output/[filename]_checkpoint.json`
- Checkpoint file contains:
  - All completed segments with timestamps and text
  - Current position in the audio
  - File metadata (duration, language, etc.)
- Automatically cleans up checkpoint file when transcription completes

### Resume from Checkpoint
1. If transcription is interrupted (Ctrl+C, crash, etc.)
2. Run `python main.py` again
3. Script automatically detects checkpoint and resumes
4. Progress bar starts from last saved position

### View Live Progress
Open the checkpoint file while transcription is running:
```json
{
  "metadata": {
    "duration": 7820.0,
    "language": "en",
    "file": "..."
  },
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "..."},
    {"start": 5.2, "end": 10.8, "text": "..."}
  ]
}
```

## Troubleshooting

### Common Issues

**1. No ffmpeg found**
- Install ffmpeg using instructions above
- Verify with: `ffmpeg -version`

**2. GPU/CUDA Issues**
- Set `FORCE_DEVICE = "cpu"` in main.py
- CPU mode works perfectly with faster-whisper

**3. Out of Memory**
- Use smaller model (`tiny` or `base`)
- Ensure checkpoint system is enabled for long files

**4. Poor Transcription Quality**
- Use larger model (`small` or `medium`)
- Set correct language with `FORCE_LANGUAGE`
- Ensure audio quality is good

### Performance Tips

- **Long Videos (1+ hours)**: Use faster-whisper on CPU with checkpoints enabled
- **Best CPU Performance**: Use `small` model with faster-whisper
- **Quality Focus**: Use `medium` or `large` models
- **Speed Focus**: Use `tiny` or `base` models

