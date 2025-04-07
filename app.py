import time
import gradio as gr
import numpy as np
import torch
import requests
import json
import os
from pathlib import Path
from gradio_webrtc import WebRTC, ReplyOnPause, AdditionalOutputs, get_turn_credentials
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from kokoro_onnx import Kokoro
import warnings
warnings.filterwarnings("ignore")

# == Configuration ==
SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.92
MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS = 250, 100
PAUSE_THRESHOLD = 1.0
SESSION_TIMEOUT, CONCURRENCY_LIMIT = 270000, 30
last_audio_timestamp = 0

# == VAD Model ==
def load_vad_model():
    print("Loading VAD model...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
    return vad_model, utils[0]

vad_model, get_speech_timestamps = load_vad_model()

def should_play_new_audio():
    global last_audio_timestamp
    current_time = time.time()
    if current_time - last_audio_timestamp < PAUSE_THRESHOLD:
        return False
    last_audio_timestamp = current_time
    return True

def check_vad(audio_array):
    try:
        # Handle stereo to mono
        audio_check = audio_array.copy()
        if len(audio_check.shape) > 1:
            if audio_check.shape[0] == 1: audio_check = audio_check.squeeze(0)
            else: audio_check = np.mean(audio_check, axis=0)
        
        # Normalize for VAD
        audio_check = audio_check.astype(np.float32) / 32768.0
        
        # Chunk size adjustment
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

# == STT Model (Whisper) ==
def load_whisper_model():
    print("Loading Whisper model...")
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo", # distil-whisper/distil-large-v2
        torch_dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"attn_implementation": attn_implementation}
    )

asr_pipeline = load_whisper_model()

def transcribe_audio(audio_array):
    try:
        # Ensure audio is properly formatted
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Process with Whisper
        result = asr_pipeline(
            audio_array,
            chunk_length_s=10,
            batch_size=8,
            return_timestamps=False
        )
        
        text = result["text"].strip()
        if not text:
            print("[DEBUG] Empty transcription")
            return False
            
        print(f"[TRANSCRIPT] {text}")
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
        return False

# == LLM Integration (Simple, Non-Streaming) ==
def send_to_llm(prompt, conversation=None, model="ollama/mistral"):
    if conversation is None:
        conversation = []
    system_prompt = "You are a voice assistant. Please respond concisely."
    
    # Convert Gradio messages to LLM format
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation:
        if isinstance(msg["content"], (str, bool, int, float)):
            llm_messages.append({"role": msg["role"], "content": str(msg["content"])})
    
    base_url = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:5000/v1')
    api_key = os.getenv('OPENAI_API_KEY', 'sk-111111111111111111111111111111111111111111111111')

    url = f"{base_url}/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    messages = llm_messages + [{"role": "user", "content": prompt}]
    data = {'model': model, 'messages': messages, 'stream': False, 'max_tokens': 2000}
    
    try:
        print(f"\nSending request to LLM...")
        start_time = time.time()
        
        # Long timeout to handle very slow responses
        response = requests.post(url, headers=headers, json=data, timeout=600)  # 10-minute timeout
        response.raise_for_status()
        
        result = response.json()
        response_text = result['choices'][0]['message']['content']
        
        print(f"LLM response received in {time.time() - start_time:.1f} seconds")
        return response_text
    except requests.exceptions.Timeout:
        print(f"\nLLM request timed out after {time.time() - start_time:.1f} seconds")
        return "I'm sorry, the response took too long. Please try again with a simpler question."
    except Exception as e:
        print(f"\nLLM Error: {e}")
        return f"Sorry, I encountered an error: {str(e)[:100]}"

# == TTS Integration ==
_kokoro_instance = None

def initialize_tts():
    global _kokoro_instance
    if _kokoro_instance is not None:
        return True
        
    try:
        cache_dir = Path(os.getenv('LOCALAPPDATA', str(Path.home()))) / 'kokoro'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        urls = {
            'kokoro-v1.0.onnx': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx',
            'voices-v1.0.bin': 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'
        }
        
        for f, u in urls.items():
            file_path = cache_dir / f
            if not file_path.exists():
                print(f"Downloading {f}...")
                file_path.write_bytes(requests.get(u).content)
        
        threads = str(os.cpu_count() or 4)
        os.environ.update({
            "ONNXRUNTIME_PROVIDER": "CPUExecutionProvider",
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads
        })
        
        _kokoro_instance = Kokoro(
            str(cache_dir / "kokoro-v1.0.onnx"),
            str(cache_dir / "voices-v1.0.bin")
        )
        return True
    except Exception as e:
        print(f"TTS initialization failed: {e}")
        return False

def text_to_speech(text, voice="bf_emma"):
    if not initialize_tts():
        return None, None
        
    try:
        return _kokoro_instance.create(text, voice=voice, speed=1.0, lang="en-us")
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None, None

# == Main Pipeline ==
def process_pipeline(audio, conversation):
    if audio is None:
        yield AdditionalOutputs(conversation)
        return
    
    sampling_rate, audio_array = audio
    print(f"[DEBUG] Audio shape: {audio_array.shape}")
    
    if not should_play_new_audio():
        return
    
    if not check_vad(audio_array):
        print("[DEBUG] VAD rejected audio")
        return
    
    print("[DEBUG] VAD detected speech")
    
    # Hotfix: replace existing audio with placeholders to stop playback
    previous_messages = []
    for msg in conversation:
        if isinstance(msg["content"], gr.Audio):
            previous_messages.append({
                "role": msg["role"],
                "content": "" 
            })
        else:
            previous_messages.append(msg)
    
    # Clear and restore conversation without audio elements
    conversation.clear()
    conversation.extend(previous_messages)
    yield AdditionalOutputs(conversation)
    
    # Transcribe speech to text
    transcript = transcribe_audio(audio_array)
    if not transcript:
        return
    
    # Add user message with transcript
    conversation.append({
        "role": "user", 
        "content": transcript
    })
    yield AdditionalOutputs(conversation)
    
    # Add a placeholder message while processing
    conversation.append({
        "role": "assistant",
        "content": "Thinking... (this may take a while)"
    })
    yield AdditionalOutputs(conversation)
    
    # Get LLM response (this might take a long time)
    llm_response = send_to_llm(transcript, conversation[:-1])  # Exclude the placeholder
    if not llm_response:
        conversation[-1]["content"] = "Sorry, I couldn't generate a response."
        yield AdditionalOutputs(conversation)
        return
    
    # Update with the actual response
    conversation[-1]["content"] = llm_response
    yield AdditionalOutputs(conversation)
    
    # Generate speech from response
    print("Generating TTS...")
    audio_samples, sample_rate = text_to_speech(llm_response)
    if audio_samples is None:
        print("[DEBUG] TTS generation failed")
        return
    
    # Add audio response
    audio_data = (sample_rate, (audio_samples * 32767).astype(np.int16))
    conversation.append({
        "role": "assistant", 
        "content": gr.Audio(
            value=audio_data,
            show_label=False,
            autoplay=True
        )
    })
    yield AdditionalOutputs(conversation)

# == Gradio Interface ==
def build_interface():
    rtc_configuration = None
    if hasattr(gr, "get_space") and gr.get_space():
        rtc_configuration = get_turn_credentials(method="twilio")
    
    with gr.Blocks(title="FlexVoice") as demo:
        gr.Markdown("# 🗣️ FlexVoice - Local Voice Chat Pipeline")
        
        with gr.Row():
            with gr.Column(scale=4):
                conversation = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                    bubble_full_width=False
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
            ReplyOnPause(process_pipeline, input_sample_rate=SAMPLE_RATE),
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
    print("Starting FlexVoice...")
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    )
