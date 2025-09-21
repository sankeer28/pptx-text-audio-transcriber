# 🎙️ PPTX Text and Audio Transcriber

Extract text and transcribe audio from PowerPoint presentations using OpenAI Whisper.

## Features

- 📝 **Text Extraction**: Extracts all text content from PowerPoint slides
- 🎤 **Audio Transcription**: Uses OpenAI Whisper to transcribe embedded audio files (WAV, MP3, M4A)
- ⚡ **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- 🎯 **Multiple Models**: Supports various Whisper model sizes (tiny, base, small, medium, large)
- 📊 **Progress Tracking**: Real-time progress bars during processing
- 🔧 **Configurable**: Easy-to-modify settings for performance and quality tuning

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster processing)

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

### 4. CUDA Setup (Optional but Recommended)

For GPU acceleration, install CUDA and compatible PyTorch:

#### Windows:
1. **Install CUDA Toolkit**:
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Choose version 11.8 or 12.1 (recommended)
   - Follow the installer instructions

2. **Install cuDNN**:
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

3. **Install CUDA-enabled PyTorch**:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### macOS/Linux:
```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Verify Installation

Test CUDA availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Usage

### 1. Prepare Your Files

- Create a `presentations` folder in the project directory
- Place your PowerPoint files (.pptx) in the `presentations` folder

### 2. Run the Extractor

```bash
python main.py
```

### 3. Check Results

- Extracted content will be saved in the `output` folder
- Each PowerPoint file generates a corresponding `.txt` file with:
  - All slide text content
  - Transcribed audio content
  - Processing metadata

## Configuration

Edit the configuration settings at the top of `main.py`:

### Folder Settings
```python
PPTX_FOLDER = "presentations"   # Input folder
OUTPUT_FOLDER = "output"        # Output folder
```

### Whisper Model Settings
```python
WHISPER_MODEL = "base"        # Options: "tiny", "base", "small", "medium", "large"
FORCE_LANGUAGE = "en"         # Force language ("en", "es", "fr", etc.) or None for auto-detect
```

### Performance Settings
```python
FORCE_DEVICE = None           # Options: None (auto), "cpu", "cuda"
USE_HALF_PRECISION = False    # Enable fp16 for 30-50% speed boost (GPU only)
GPU_BEST_OF = 3              # Higher = more accurate, slower
GPU_BEAM_SIZE = 3            # Beam search size
```

### Quality Settings
```python
TEMPERATURE = 0.0             # 0.0 = deterministic, 0.1-1.0 = more creative
ENABLE_WORD_TIMESTAMPS = True # Get word-level timing data
```

## Model Size Guide

| Model  | Size | Speed | Quality | VRAM Usage |
|--------|------|-------|---------|------------|
| tiny   | 39MB | Fastest | Good | ~1GB |
| base   | 74MB | Fast | Better | ~1GB |
| small  | 244MB | Medium | Good | ~2GB |
| medium | 769MB | Slow | Very Good | ~5GB |
| large  | 1550MB | Slowest | Best | ~10GB |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Use a smaller Whisper model (`tiny` or `base`)
- Set `USE_HALF_PRECISION = True`
- Reduce `GPU_BEST_OF` and `GPU_BEAM_SIZE`

**2. No Audio Files Found**
- Ensure audio is embedded in PowerPoint (not linked)
- Supported formats: WAV, MP3, M4A

**3. Installation Issues**
- Ensure Python 3.8+ is installed
- Try installing dependencies one by one
- Use virtual environment to avoid conflicts

**4. Poor Transcription Quality**
- Use larger Whisper model (`medium` or `large`)
- Set correct language with `FORCE_LANGUAGE`
- Increase `GPU_BEST_OF` for better accuracy

### Performance Tips

- **GPU Users**: Use `base` or `small` models for best speed/quality balance
- **CPU Users**: Stick with `tiny` or `base` models
- **Large Files**: Process in batches to avoid memory issues
- **Quality Focus**: Use `medium` or `large` models with higher beam size

