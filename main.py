import os
import zipfile
import shutil
import re
import warnings
from pptx import Presentation
import whisper
import torch
from tqdm import tqdm

# Suppress CUDA/Triton warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# ⚙️ CONFIGURATION SETTINGS - Edit these for easy customization
# 📂 Folder Settings
PPTX_FOLDER = "presentations"   # input folder
OUTPUT_FOLDER = "output"        # output folder

# 🎯 Whisper Model Settings
WHISPER_MODEL = "base"        # Options: "tiny", "base", "small", "medium", "large"
FORCE_LANGUAGE = "en"           # Force language to prevent mixing (None for auto-detect)

# ⚡ Performance Settings
FORCE_DEVICE = None             # Options: None (auto), "cpu", "cuda" (force specific device)
USE_HALF_PRECISION = False       # fp16 for 30-50% speed boost (minimal accuracy loss)
GPU_BEST_OF = 3                 # Decoding attempts on GPU (higher = more accurate, slower)
GPU_BEAM_SIZE = 3               # Beam search size on GPU
CPU_BEST_OF = 3                 # Decoding attempts on CPU
CPU_BEAM_SIZE = 3               # Beam search size on CPU

# 🎚️ Quality Settings
TEMPERATURE = 0.0               # 0.0 = deterministic, 0.1-1.0 = more creative
ENABLE_WORD_TIMESTAMPS = True   # Get word-level timing data

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ⚡ Load Whisper model with optimal device selection
def get_optimal_device():
    # Check if user forced a specific device
    if FORCE_DEVICE:
        if FORCE_DEVICE == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            return "cpu"
        print(f"🔧 Forced device: {FORCE_DEVICE}")
        return FORCE_DEVICE

    # Auto-detect best device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        return "cuda"
    else:
        print("No GPU detected, using CPU")
        return "cpu"

device = get_optimal_device()
print(f"Loading Whisper model on {device}...")

# Use GPU for main model if available, fallback to CPU for specific operations
model = whisper.load_model(WHISPER_MODEL, device=device)

def extract_text_from_pptx(pptx_path):
    """Extract all text from slides in a PPTX."""
    prs = Presentation(pptx_path)
    texts = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    slide_texts.append(text)
        if slide_texts:
            texts.append(f"--- Slide {slide_num} ---\n" + "\n".join(slide_texts))
    return "\n\n".join(texts)

def extract_audio_from_pptx(pptx_path, temp_dir):
    """Extract embedded audio files from PPTX (wav, mp3, m4a) with proper ordering."""
    audio_files = []
    with zipfile.ZipFile(pptx_path, "r") as zip_ref:
        media_files = []
        for file in zip_ref.namelist():
            if file.startswith("ppt/media/") and file.lower().endswith((".wav", ".mp3", ".m4a")):
                media_files.append(file)

        # Sort by the numeric part in filename (media1, media2, etc.)
        def get_media_number(filename):
            match = re.search(r'media(\d+)', filename)
            return int(match.group(1)) if match else 0

        media_files.sort(key=get_media_number)

        for file in media_files:
            extracted_path = os.path.join(temp_dir, os.path.basename(file))
            with open(extracted_path, "wb") as f:
                f.write(zip_ref.read(file))
            audio_files.append(extracted_path)

    return audio_files

def transcribe_audio(audio_files):
    """Run Whisper transcription with hybrid CPU/GPU optimization and progress bar."""
    transcripts = []

    # Create progress bar
    progress_bar = tqdm(
        audio_files,
        desc="🎤 Transcribing audio",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    for audio_path in progress_bar:
        # Update progress bar description with current file
        filename = os.path.basename(audio_path)
        media_match = re.search(r'media(\d+)', filename)
        media_num = media_match.group(1) if media_match else "unknown"

        progress_bar.set_description(f"🎤 Processing Audio {media_num}")

        try:
            # Try GPU first for main transcription
            if device == "cuda":
                result = model.transcribe(
                    audio_path,
                    language=FORCE_LANGUAGE,
                    task="transcribe",
                    temperature=TEMPERATURE,
                    best_of=GPU_BEST_OF,
                    beam_size=GPU_BEAM_SIZE,
                    word_timestamps=ENABLE_WORD_TIMESTAMPS,
                    fp16=USE_HALF_PRECISION
                )
            else:
                # CPU optimization
                result = model.transcribe(
                    audio_path,
                    language=FORCE_LANGUAGE,
                    task="transcribe",
                    temperature=TEMPERATURE,
                    best_of=CPU_BEST_OF,
                    beam_size=CPU_BEAM_SIZE,
                    word_timestamps=ENABLE_WORD_TIMESTAMPS,
                    fp16=False  # Always False on CPU
                )

        except Exception as e:
            progress_bar.write(f"⚠️ GPU failed for {filename}, falling back to CPU...")
            # Fallback to CPU if GPU fails
            cpu_model = whisper.load_model(WHISPER_MODEL, device="cpu")
            result = cpu_model.transcribe(
                audio_path,
                language=FORCE_LANGUAGE,
                temperature=TEMPERATURE
            )

        transcripts.append(f"--- Audio {media_num} Transcript ---\n{result['text'].strip()}")

        # Clear GPU cache periodically if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    progress_bar.close()
    return "\n\n".join(transcripts)

def process_pptx(pptx_path):
    """Process one PowerPoint: text + audio → output TXT file."""
    print(f"\n📑 Processing {pptx_path}...")
    base_name = os.path.splitext(os.path.basename(pptx_path))[0]
    temp_dir = os.path.join(OUTPUT_FOLDER, base_name + "_media")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Slide text
        text_content = extract_text_from_pptx(pptx_path)

        # Embedded audio
        audio_files = extract_audio_from_pptx(pptx_path, temp_dir)
        transcript = transcribe_audio(audio_files) if audio_files else ""

        # Combine output with improved organization
        final_output = []
        if text_content:
            final_output.append("### PowerPoint Slide Content (In Order) ###\n" + text_content)
        if transcript:
            final_output.append("\n### Audio Transcripts (In Chronological Order) ###\n" + transcript)

        # Add summary note about ordering
        if text_content and transcript:
            final_output.append("\n### Note ###\nAudio transcripts are generated by OpenAI Whisper.")

        # Save result
        output_path = os.path.join(OUTPUT_FOLDER, base_name + ".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(final_output))

        print(f"✅ Saved results to {output_path}")

    finally:
        # Cleanup extracted media
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    pptx_files = [f for f in os.listdir(PPTX_FOLDER) if f.lower().endswith(".pptx")]
    if not pptx_files:
        print("⚠️ No .pptx files found in", PPTX_FOLDER)
        return
    for file in pptx_files:
        process_pptx(os.path.join(PPTX_FOLDER, file))

if __name__ == "__main__":
    main()
