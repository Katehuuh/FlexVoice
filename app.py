# Step 1: Create individual components
# Module 1 - STT 'insanely-fast-whisper', base no user interaction, using chunks instead of WebRTC for audio capture.
import torch
import numpy as np
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline
import time
import os
import onnxruntime as ort

ort.set_default_logger_severity(3)
SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.9
MIN_SILENCE = 0.8
MAX_DURATION = 10.0
TEMP_FILE = "temp_speech.wav"

# Global state (required for real-time processing)
audio_chunks = []
last_process_time = time.time()

def initialize_models():
    """Initialize and return VAD and ASR models"""
    print("Loading VAD model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=True
    )
    get_speech_timestamps = utils[0]

    print("Loading Whisper model...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v2",
        torch_dtype=torch.float16,
        device="cuda:0"
    )
    return vad_model, get_speech_timestamps, asr_pipeline

def detect_speech(audio, vad_model, get_speech_timestamps):
    """Detect speech in audio segment"""
    # Ensure audio length is multiple of 512
    if len(audio) % 512 != 0:
        audio = audio[:(len(audio) // 512) * 512]
    
    try:
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio).float(),
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=SPEECH_THRESHOLD,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100
        )
        return bool(speech_timestamps)
    except Exception as e:
        print(f"VAD error (non-critical): {e}")
        return False

def process_audio(audio, vad_model, get_speech_timestamps, asr_pipeline):
    """Process audio chunk and return transcription if available"""
    global audio_chunks, last_process_time
    
    if detect_speech(audio, vad_model, get_speech_timestamps):
        audio_chunks.append(audio)
        print("Speech detected...")
        return None

    if not audio_chunks:
        return None

    silence_duration = time.time() - last_process_time
    current_duration = len(np.concatenate(audio_chunks)) / SAMPLE_RATE
    
    if silence_duration >= MIN_SILENCE or current_duration >= MAX_DURATION:
        print(f"\nProcessing segment (duration: {current_duration:.1f}s)")
        full_audio = np.concatenate(audio_chunks)
        sf.write(TEMP_FILE, full_audio, SAMPLE_RATE)
        
        try:
            result = asr_pipeline(TEMP_FILE)
            transcription = result["text"].strip()
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"ASR error: {e}")
            transcription = None
        
        audio_chunks.clear()
        last_process_time = time.time()
        return transcription
    return None

def setup_microphone():
    """Initialize microphone with proper settings"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=SAMPLE_RATE)
    
    with mic as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Start speaking...")
    
    return recognizer, mic

def main():
    """Main function for real-time speech processing"""
    # Initialize models
    vad_model, get_speech_timestamps, asr_pipeline = initialize_models()
    recognizer, mic = setup_microphone()
    
    def audio_callback(_, audio):
        try:
            data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            data = data.astype(np.float32) / 32768.0
            process_audio(data, vad_model, get_speech_timestamps, asr_pipeline)
        except Exception as e:
            print(f"Error in callback: {e}")
    
    stop_listening = recognizer.listen_in_background(mic, audio_callback)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()

# Module 2 - self-host openai API with history msg
import os
import requests
import json

def SendMessage(prompt, history=None, model="ollama/mistral"):
    """Send message to LLM and get response"""
    if history is None:
        history = []
    
    base_url = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:5000/v1')
    api_key = os.getenv('OPENAI_API_KEY', 'sk-111111111111111111111111111111111111111111111111')
    
    url = f"{base_url}/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    messages = history + [{"role": "user", "content": prompt}]
    data = {'model': model, 'messages': messages, 'stream': True}
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        full_response = ""
        print("\nAssistant: ", end='', flush=True)
        
        for line in response.iter_lines():
            if not line or not line.startswith(b'data: '): continue
            if line == b'data: [DONE]': break
            
            try:
                content = json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                if content:
                    print(content, end='', flush=True)
                    full_response += content
            except: continue
        
        print()  # New line
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response}
        ])
        return full_response, history
        
    except Exception as e:
        print(f"\nError: {e}")
        return None, history

def main():
    history = []
    while True:
        msg = input("\nYou: ")
        if msg.lower() in ['exit', 'quit']: break
        response, history = SendMessage(msg, history)

if __name__ == "__main__":
    main()
  
# Module 3 - TTS kokoro onnx
from pathlib import Path
import os, requests, numpy as np
from typing import Tuple, Optional
from kokoro_onnx import Kokoro

_kokoro_instance = None

def DownloadKokoroTTS() -> bool:
    try:
        cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
        cache_dir.mkdir(parents=True, exist_ok=True)
        urls = {'kokoro-v1.0.onnx': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx',
               'voices-v1.0.bin': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'}
        return all(Path(cache_dir/f).exists() or Path(cache_dir/f).write_bytes(requests.get(u).content) for f,u in urls.items())
    except Exception as e: print(f"Download failed: {e}"); return False

def SendKokoroTTS(text: str, voice: str = "bf_emma") -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Convert text to speech using cached Kokoro TTS model
    Args:
        text: Text to convert to speech
        voice: Voice ID to use (default: bf_emma - fastest)
    
    Available Voices:
        Female English:
        - af_bella: Calm, clear voice
        - af_sarah: Professional narrator
        - af_nicole: Soft, mysterious tone
        - af_sky: Young, energetic voice
        
        Male English:
        - am_adam: Deep, authoritative
        - am_michael: Friendly, conversational
        - bm_george: British accent
        - bm_lewis: Mature, professional
        
        Female Other:
        - bf_emma: Fast, clear (best for performance)
        - bf_isabella: Elegant tone
    
    Hidden Options:
        speed: 0.5-2.0 (default 1.0) - Adjust speech rate
        lang: "en-us" (default) - Language code
        pitch: 0.5-2.0 (hidden) - Voice pitch adjustment
        energy: 0.5-2.0 (hidden) - Voice energy level
    """
    global _kokoro_instance
    try:
        if _kokoro_instance is None:
            if not DownloadKokoroTTS(): return None, None
            threads = str(os.cpu_count() or 4)
            os.environ.update({
                "ONNXRUNTIME_PROVIDER": "CPUExecutionProvider",
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads
            })
            cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
            _kokoro_instance = Kokoro(str(cache_dir/"kokoro-v1.0.onnx"), str(cache_dir/"voices-v1.0.bin"))
        return _kokoro_instance.create(text, voice=voice, speed=1.0, lang="en-us")
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None, None

if __name__ == "__main__":
    text = "This is a test of the TTS system."
    samples, sr = SendKokoroTTS(text)
    if samples is not None:
        import soundfile as sf
        sf.write("test.wav", samples, sr)
        print("Audio saved to test.wav")


# Step 2: Module part 1+2+3 into a single CMD app; no WebRTC, using chunks for detecting silence:
import os
import time
import torch
import numpy as np
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline
import requests
import json
from pathlib import Path
from kokoro_onnx import Kokoro
import onnxruntime as ort

ort.set_default_logger_severity(3)
SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.9
MIN_SILENCE = 0.8
MAX_DURATION = 10.0
TEMP_FILE = "temp_speech.wav"

# Global cache for models
_vad_model = None
_asr_pipeline = None
_kokoro_instance = None
audio_chunks = []
last_process_time = time.time()

# STT Functions
def initialize_stt():
    """Initialize VAD and ASR models"""
    global _vad_model, _asr_pipeline
    
    print("Loading VAD model...")
    _vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=True
    )
    get_speech_timestamps = utils[0]

    print("Loading Whisper model...")
    _asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v2",
        torch_dtype=torch.float16,
        device="cuda:0"
    )
    return get_speech_timestamps

def detect_speech(audio, get_speech_timestamps):
    """Detect speech in audio segment"""
    if len(audio) % 512 != 0:
        audio = audio[:(len(audio) // 512) * 512]
    
    try:
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio).float(),
            _vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=SPEECH_THRESHOLD,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100
        )
        return bool(speech_timestamps)
    except Exception as e:
        print(f"VAD error: {e}")
        return False

def process_audio(audio, get_speech_timestamps):
    """Process audio chunk and return transcription if available"""
    global audio_chunks, last_process_time
    
    if detect_speech(audio, get_speech_timestamps):
        audio_chunks.append(audio)
        print("Speech detected...")
        return None

    if not audio_chunks:
        return None

    silence_duration = time.time() - last_process_time
    current_duration = len(np.concatenate(audio_chunks)) / SAMPLE_RATE
    
    if silence_duration >= MIN_SILENCE or current_duration >= MAX_DURATION:
        full_audio = np.concatenate(audio_chunks)
        sf.write(TEMP_FILE, full_audio, SAMPLE_RATE)
        
        try:
            result = _asr_pipeline(TEMP_FILE)
            transcription = result["text"].strip()
            print(f"Transcription: {transcription}")
            audio_chunks.clear()
            last_process_time = time.time()
            return transcription
        except Exception as e:
            print(f"ASR error: {e}")
            audio_chunks.clear()
            return None
    return None

# LLM Functions
def send_message(prompt, history=None):
    """Send message to LLM and get response"""
    if history is None:
        history = []
    
    base_url = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:5000/v1')
    api_key = os.getenv('OPENAI_API_KEY', 'sk-111111111111111111111111111111111111111111111111')
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            json={'model': 'ollama/mistral', 'messages': history + [{"role": "user", "content": prompt}], 'stream': True},
            stream=True
        )
        
        full_response = ""
        for line in response.iter_lines():
            if line.startswith(b'data: ') and line != b'data: [DONE]':
                try:
                    content = json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)
                        full_response += content
                except: continue
        
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_response}
        ])
        return full_response, history
    except Exception as e:
        print(f"LLM error: {e}")
        return None, history

# TTS Functions
def initialize_tts():
    """Initialize TTS if not already done"""
    global _kokoro_instance
    if _kokoro_instance is None:
        cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        urls = {
            'kokoro-v1.0.onnx': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx',
            'voices-v1.0.bin': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'
        }
        
        for fname, url in urls.items():
            fpath = cache_dir / fname
            if not fpath.exists():
                fpath.write_bytes(requests.get(url).content)
        
        threads = str(os.cpu_count() or 4)
        os.environ.update({
            "ONNXRUNTIME_PROVIDER": "CPUExecutionProvider",
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads
        })
        _kokoro_instance = Kokoro(
            str(cache_dir/"kokoro-v1.0.onnx"),
            str(cache_dir/"voices-v1.0.bin")
        )

def generate_speech(text, voice="bf_emma"):
    """Generate speech from text"""
    initialize_tts()
    try:
        return _kokoro_instance.create(text, voice=voice, speed=1.0, lang="en-us")
    except Exception as e:
        print(f"TTS error: {e}")
        return None, None

def main():
    get_speech_timestamps = initialize_stt()
    
    # Create a dictionary to store state that needs to be accessed in the callback
    state = {
        'history': []
    }
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=SAMPLE_RATE)
    
    def audio_callback(_, audio):
        try:
            data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            data = data.astype(np.float32) / 32768.0
            
            transcription = process_audio(data, get_speech_timestamps)
            if transcription:
                response, state['history'] = send_message(transcription, state['history'])
                if response:
                    samples, sr = generate_speech(response)
                    if samples is not None:
                        sf.write("response.wav", samples, sr)
        except Exception as e:
            print(f"Error: {e}")
    
    with mic as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Start speaking...")
    
    stop_listening = recognizer.listen_in_background(mic, audio_callback)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()

# Step 3: Modify module 1 (echo speech for testing) to support the Gradio UI. Audio chunk manipulation is replaced by ReplyOnPause from WebRTC(+VAD), inspired by the Gradio Ultravox approach, not Moshi or GLM-4-Voice.
import time
import gradio as gr
import numpy as np
import torch
from gradio_webrtc import WebRTC, ReplyOnPause, AdditionalOutputs, get_turn_credentials

SAMPLE_RATE = 16000
PAUSE_THRESHOLD = 1.0
SPEECH_THRESHOLD = 0.92
MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS = 250, 100
SESSION_TIMEOUT, CONCURRENCY_LIMIT = 600, 30

def load_vad_model():
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=True
        )
        return vad_model, utils[0]
    except Exception as e:
        print(f"VAD loading error: {e}")
        raise

vad_model, get_speech_timestamps = load_vad_model()
last_audio_timestamp = 0

def should_play_new_audio():
    global last_audio_timestamp
    current_time = time.time()
    if current_time - last_audio_timestamp < PAUSE_THRESHOLD:
        return False
    last_audio_timestamp = current_time
    return True

def check_vad(audio_array):
    try: # Handle stereo to mono
        audio_check = audio_array.copy()
        if len(audio_check.shape) > 1:
            if audio_check.shape[0] == 1:
                audio_check = audio_check.squeeze(0)
            else:
                audio_check = np.mean(audio_check, axis=0)
        
        audio_check = audio_check.astype(np.float32) / 32768.0 # Normalize for VAD
        
        if len(audio_check) % 512 != 0: # Chunk size adjustment
            audio_check = audio_check[:(len(audio_check) // 512) * 512]
        
        if len(audio_check) == 0:
            return False
        
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio_check),
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=SPEECH_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS
        )
        return bool(speech_timestamps)
    except Exception as e:
        print(f"VAD error: {e}")
        return False

def transcribe_webrtc(audio, conversation):
    if audio is None:
        yield AdditionalOutputs(conversation)
        return
    
    sampling_rate, audio_array = audio
    print(f"[DEBUG] RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}, shape: {audio_array.shape}, format: {audio_array.dtype}")
    
    if not should_play_new_audio():
        return
    
    if not check_vad(audio_array):
        print("[DEBUG] VAD rejected audio")
        return
    
    print("[DEBUG] VAD detected speech")
    
    previous_messages = []  # Store previous messages as text
    for msg in conversation:
        if isinstance(msg["content"], gr.Audio):
            previous_messages.append({
                "role": msg["role"],
                "content": f"[Audio Message {len(previous_messages) + 1}]" # Replace previous audio widget with a placeholder to pause playback and prevent overlapping.
            })
        else:
            previous_messages.append(msg)
    
    conversation.clear()
    conversation.extend(previous_messages)
    
    conversation.append({  # Add assistant's response - keeping original audio handling
        "role": "user",
        "content": gr.Audio(
            value=(sampling_rate, audio_array.squeeze()),
            show_label=False,
            autoplay=False
        )
    })
    yield AdditionalOutputs(conversation)
    
    conversation.append({ 
        "role": "assistant",
        "content": gr.Audio(
            value=(sampling_rate, audio_array.squeeze()),
            show_label=False,
            autoplay=True
        )
    })
    yield AdditionalOutputs(conversation)

def build_interface():
    rtc_configuration = None
    if hasattr(gr, "get_space") and gr.get_space():
        rtc_configuration = get_turn_credentials(method="twilio")
    
    with gr.Blocks() as demo:
        gr.Markdown("# Voice Echo Test - With VAD")
        
        with gr.Row():
            with gr.Column(scale=4):
                conversation = gr.Chatbot(
                    label="Voice History",
                    height=400,
                    type="messages"
                )
            
            with gr.Column(scale=1):
                audio_webrtc = WebRTC(
                    rtc_configuration=rtc_configuration,
                    label="Voice Stream",
                    mode="send",
                    modality="audio"
                )
                clear_btn = gr.Button("Clear History")
        
        clear_btn.click(lambda: [], outputs=[conversation])
        
        audio_webrtc.stream(
            ReplyOnPause(
                transcribe_webrtc,
                input_sample_rate=SAMPLE_RATE
            ),
            inputs=[audio_webrtc, conversation],
            outputs=[audio_webrtc],
            time_limit=SESSION_TIMEOUT,
            concurrency_limit=CONCURRENCY_LIMIT
        )
        
        audio_webrtc.on_additional_outputs(
            lambda conv: conv,
            outputs=[conversation],
            queue=False
        )
    
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="localhost",
        server_port=7861,
        show_error=True
    )

# Step 4: Modify Gradio to replace the echo testing with STT+LLM+TTS.
import time
import gradio as gr
import numpy as np
import torch
from gradio_webrtc import WebRTC, ReplyOnPause, AdditionalOutputs, get_turn_credentials
from transformers import pipeline, WhisperProcessor
from transformers.utils import is_flash_attn_2_available
import os
import requests
import json
from pathlib import Path
from kokoro_onnx import Kokoro

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.92
MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS = 250, 100
SESSION_TIMEOUT, CONCURRENCY_LIMIT = 600, 30

# ===== VAD + Whisper Components =====

def load_stt_models():
    """
    Load VAD and Whisper models with optimizations.
    
    Returns:
        tuple: (vad_model, get_speech_timestamps, asr_pipeline)
    """
    try:
        # Load Silero VAD
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=True
        )
        get_speech_timestamps = utils[0]
        
        # Determine attention implementation
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        
        # Initialize Whisper pipeline with optimizations
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo", # or distil-whisper/distil-large-v2
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={
                "attn_implementation": attn_implementation  # Only include attention optimization
            }
        )
        
        return vad_model, get_speech_timestamps, asr_pipeline
    
    except Exception as e:
        print(f"STT Model loading error: {e}")
        raise
# Load models globally
vad_model, get_speech_timestamps, asr_pipeline = load_stt_models()

def check_vad(audio_array):
    try:
        audio_check = audio_array.copy()
        if len(audio_check.shape) > 1:
            audio_check = np.mean(audio_check, axis=0) if audio_check.shape[0] != 1 else audio_check.squeeze(0)
        
        audio_check = audio_check.astype(np.float32) / 32768.0
        if len(audio_check) % 512 != 0:
            audio_check = audio_check[:(len(audio_check) // 512) * 512]
        
        if len(audio_check) == 0:
            return False
        
        speech_timestamps = get_speech_timestamps(
            torch.from_numpy(audio_check),
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=SPEECH_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS
        )
        return bool(speech_timestamps)
    except Exception as e:
        print(f"VAD error: {e}")
        return False


def transcribe_with_whisper(audio_array):
    """
    Transcribe audio using Whisper with insanely-fast-whisper optimizations.
    
    Args:
        audio_array: NumPy array or Torch tensor of audio data
        
    Returns:
        str or False: Transcribed text if successful, False otherwise
    """
    try:
        # Convert Torch tensor to NumPy array if necessary
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        
        # Ensure audio is mono (single channel)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Use the ASR pipeline with chunking and batching
        result = asr_pipeline(
            audio_array,
            chunk_length_s=10,  # Process audio in 10-second chunks
            batch_size=8,       # Process up to 8 chunks in parallel
            return_timestamps=False  # Set to True if you need timestamps
        )
        
        # Extract and clean the transcribed text
        text = result["text"].strip()
        if not text:
            print("[DEBUG] Whisper: Empty transcription")
            return False
        print(f"[DEBUG] Whisper: {text}")
        return text
    
    except Exception as e:
        print(f"Whisper error: {e}")
        return False

# ===== LLM Component =====
def send_to_llm(prompt, conversation=None, model="ollama/mistral"):
    """Send message to LLM and get response"""
    if conversation is None:
        conversation = []
    
    # Convert Gradio messages to LLM format
    llm_messages = []
    for msg in conversation:
        # Skip audio messages
        if isinstance(msg["content"], (str, bool, int, float)):
            llm_messages.append({
                "role": msg["role"],
                "content": str(msg["content"])
            })
    
    base_url = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:5000/v1')
    api_key = os.getenv('OPENAI_API_KEY', 'sk-111111111111111111111111111111111111111111111111')
    
    url = f"{base_url}/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    messages = llm_messages + [{"role": "user", "content": prompt}]
    data = {'model': model, 'messages': messages, 'stream': True}
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        full_response = ""
        print("\nAssistant: ", end='', flush=True)
        
        for line in response.iter_lines():
            if not line or not line.startswith(b'data: '): continue
            if line == b'data: [DONE]': break
            
            try:
                content = json.loads(line[6:])['choices'][0]['delta'].get('content', '')
                if content:
                    print(content, end='', flush=True)
                    full_response += content
            except: continue
        
        print()
        return full_response
        
    except Exception as e:
        print(f"\nError: {e}")
        return None

# ===== TTS Component =====
_kokoro_instance = None

def initialize_tts():
    try:
        cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
        cache_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            'kokoro-v1.0.onnx': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx',
            'voices-v1.0.bin': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'
        }
        return all(Path(cache_dir/f).exists() or Path(cache_dir/f).write_bytes(requests.get(u).content) for f,u in urls.items())
    except Exception as e:
        print(f"TTS initialization failed: {e}")
        return False

def text_to_speech(text, voice="bf_emma"):
    global _kokoro_instance
    try:
        if _kokoro_instance is None:
            if not initialize_tts():
                return None, None
            threads = str(os.cpu_count() or 4)
            os.environ.update({
                "ONNXRUNTIME_PROVIDER": "CPUExecutionProvider",
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads
            })
            cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
            _kokoro_instance = Kokoro(str(cache_dir/"kokoro-v1.0.onnx"), str(cache_dir/"voices-v1.0.bin"))
        return _kokoro_instance.create(text, voice=voice, speed=1.0, lang="en-us")
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None, None

# ===== Main Pipeline =====
def process_audio(audio_array):
    if not check_vad(audio_array):
        print("[DEBUG] VAD rejected audio")
        return False, None
    print("[DEBUG] VAD detected speech")
    
    text = transcribe_with_whisper(audio_array.squeeze())
    if not text:
        print("[DEBUG] Whisper rejected audio")
        return False, None
    
    return True, text

def process_pipeline(audio, conversation):
    if audio is None:
        yield AdditionalOutputs(conversation)
        return
    
    sampling_rate, audio_array = audio
    print(f"[DEBUG] Audio - RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}, shape: {audio_array.shape}")
    
    # STT
    is_valid, text = process_audio(audio_array)
    if not is_valid:
        return
    
    # Add user's transcribed message
    conversation.append({"role": "user", "content": text})
    yield AdditionalOutputs(conversation)
    
    # Get LLM response
    llm_response = send_to_llm(text, conversation)
    if not llm_response:
        return
    print(f"[DEBUG] LLM Response: {llm_response}")
    
    # Add text response
    conversation.append({"role": "assistant", "content": llm_response})
    yield AdditionalOutputs(conversation)
    
    # Generate speech with Kokoro TTS
    audio_samples, sample_rate = text_to_speech(llm_response)
    if audio_samples is None:
        print("[DEBUG] TTS failed")
    else:
        # Create audio data
        audio_data = (sample_rate, (audio_samples * 32767).astype(np.int16))
        # Add audio as a separate message with special handling
        conversation.append({
            "role": "assistant",
            "content": gr.Audio(
                value=audio_data,
                visible=True,
                autoplay=True,
                show_label=False
            )
        })
        yield AdditionalOutputs(conversation)
# ===== Gradio Interface =====
def build_interface():
    rtc_configuration = None
    if hasattr(gr, "get_space") and gr.get_space():
        rtc_configuration = get_turn_credentials(method="twilio")
    
    with gr.Blocks() as demo:
        gr.Markdown("# Voice Chat with VAD + Whisper + LLM + TTS")
        
        with gr.Row():
            with gr.Column(scale=4):
                conversation = gr.Chatbot(
                    label="Conversation History",
                    height=400,
                    type="messages",
                    bubble_full_width=False,
                    show_label=True
                )
            
            with gr.Column(scale=1):
                audio_webrtc = WebRTC(
                    rtc_configuration=rtc_configuration,
                    label="Voice Input",
                    mode="send",
                    modality="audio"
                )
                clear_btn = gr.Button("Clear History")
        
        clear_btn.click(lambda: [], outputs=[conversation])
        
        audio_webrtc.stream(
            fn=ReplyOnPause(
                process_pipeline,
                input_sample_rate=SAMPLE_RATE
            ),
            inputs=[audio_webrtc, conversation],
            outputs=[audio_webrtc],
            time_limit=SESSION_TIMEOUT,
            concurrency_limit=CONCURRENCY_LIMIT
        )
        
        audio_webrtc.on_additional_outputs(
            lambda conv: conv,
            outputs=[conversation],
            queue=False
        )
        
        return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        # share=True,
        show_error=True
    )
