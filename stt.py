import sounddevice as sd
import numpy as np
import time
import collections
import sys
import torch
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import wavio
import os
import google.generativeai as genai
import re # Added for markdown stripping

# --- Configuration ---
MODEL_SIZE = "Oriserve/Whisper-Hindi2Hinglish-Apex" # Using the Apex model as requested
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
TRANSCRIPTION_INTERVAL_S = 10
TRANSCRIPT_FILE = "transcript_log.txt"

# --- ANSI Color Codes ---
COLOR_GREEN = "\033[92m" # Green for User
COLOR_BLUE = "\033[94m"  # Blue for Gemini
COLOR_RESET = "\033[0m"  # Reset color

def strip_markdown(text):
    # Remove common markdown elements for cleaner terminal output
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # Code blocks
    text = re.sub(r'```.*', '', text) # Inline code blocks/fences
    text = re.sub(r'^[*-]\s+', '', text, flags=re.MULTILINE) # List items
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE) # Headers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text) # Italic
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text) # Links
    text = re.sub(r'^\s*[-*_=]+\s*$', '', text, flags=re.MULTILINE) # Horizontal rules
    return text.strip()

def colored(text, color_code):
    return f"{color_code}{text}{COLOR_RESET}"

# --- Global State ---
audio_buffer = collections.deque()

def audio_callback(indata, frames, time, status):
    """This function is called by the sounddevice stream for each new audio chunk."""
    if status:
        print(f"Audio Status: {status}", file=sys.stderr)
    audio_buffer.append(indata.copy())

# --- User-provided functions (modified for clarity) ---

def initialize_stt_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading STT model '{MODEL_SIZE}' on device '{device}'...", file=sys.stderr)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_SIZE,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_SIZE)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype, return_timestamps=True, device=device,
        generate_kwargs={"task": "transcribe", "language": "en"}
    )
    print("STT Model loaded successfully.", file=sys.stderr)
    return pipe

def transcribe_audio(pipe, filePath):
    print(f"Transcribing {filePath}...", file=sys.stderr)
    result = pipe(filePath)
    return result["text"]

def initialize_gemini():
    """Initializes the Gemini model and checks for the API key."""
    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview') # Using gemini-1.0-pro as it's a stable choice
        print("Gemini model initialized successfully.", file=sys.stderr)
        return model
    except KeyError:
        print("ERROR: GEMINI_API_KEY environment variable not found.", file=sys.stderr)
        print("Please set your API key using: $env:GEMINI_API_KEY = 'YOUR_KEY'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error initializing Gemini: {e}", file=sys.stderr)
        return None

def main():
    # 1. Initialize Models
    stt_pipe = initialize_stt_model()
    gemini_model = initialize_gemini()
    if gemini_model is None:
        sys.exit(1) # Exit if Gemini isn't configured

    # 2. Start audio capture
    print(f"\nStarting audio stream... Transcribing every {TRANSCRIPTION_INTERVAL_S} seconds.", file=sys.stderr)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback
        ):
            temp_file_path = "temp_audio.wav"

            while True:
                time.sleep(TRANSCRIPTION_INTERVAL_S)
                
                if not audio_buffer:
                    continue

                try:
                    all_chunks = np.concatenate(list(audio_buffer))
                    audio_buffer.clear()
                except ValueError:
                    continue
                
                # 3. Save audio chunk to a temporary file
                wavio.write(temp_file_path, all_chunks, SAMPLE_RATE, sampwidth=2)
                
                # 4. Transcribe the audio
                user_text = transcribe_audio(stt_pipe, temp_file_path).strip()
                
                if user_text:
                    print(colored(f"You: {user_text}", COLOR_GREEN))
                    
                    # Log the user's text
                    with open(TRANSCRIPT_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"You: {user_text}\n")
                    
                    # Get and print Gemini's response
                    print(colored("Gemini: ...", COLOR_BLUE)) # Show thinking state
                    response = gemini_model.generate_content(user_text)
                    gemini_response = strip_markdown(response.text.strip())
                    
                    print(colored(f"Gemini: {gemini_response}\n", COLOR_BLUE))
                    
                    # Log Gemini's response
                    with open(TRANSCRIPT_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"Gemini: {gemini_response}\n\n")

    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
