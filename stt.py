import torch
import ffmpeg
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def initializeModel():
    # Set device (GPU if available, otherwise CPU) and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Specify the pre-trained model ID
    model_id = "Oriserve/Whisper-Hindi2Hinglish-Apex"

    # Load the speech-to-text model with specified configurations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,        # Use appropriate precision (float16 for GPU, float32 for CPU)
        low_cpu_mem_usage=True,         # Optimize memory usage during loading
        use_safetensors=True            # Use safetensors format for better security
    )
    model.to(device)                    # Move model to specified device

    # Load the processor for audio preprocessing and tokenization
    processor = AutoProcessor.from_pretrained(model_id)

    # Create speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        return_timestamps=True,
        device=device,
        generate_kwargs={
            "task": "transcribe",       # Set task to transcription
            "language": "en"            # Specify English language
        }
    )

    return pipe

def transcribe(pipe, filePath):
    # Process audio file and print transcription
    result = pipe(filePath)               # Run inference
    return result["text"]               # Print transcribed text