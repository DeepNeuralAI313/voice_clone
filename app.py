import spaces
import torch
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import gradio as gr
import traceback
import gc
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
from huggingface_hub import snapshot_download
from tts.infer_cli import Voice_cloning_model, convert_to_wav, cut_wav


def download_weights():
    """Download model weights from HuggingFace if not already present."""
    repo_id = "mrfakename/MegaTTS3-VoiceCloning"
    weights_dir = "checkpoints"
    
    if not os.path.exists(weights_dir):
        print("Downloading model weights from HuggingFace...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=weights_dir,
            local_dir_use_symlinks=False
        )
        print("Model weights downloaded successfully!")
    else:
        print("Model weights already exist.")
    
    return weights_dir


# Download weights and initialize model
download_weights()
print("Initializing MegaTTS3 model...")
infer_pipe = Voice_cloning_model()
print("Model loaded successfully!")

def reset_model():
    """Reset the inference pipeline to recover from CUDA errors."""
    global infer_pipe
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Reinitializing MegaTTS3 model...")
        infer_pipe = Voice_cloning_model()
        print("Model reinitialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to reinitialize model: {e}")
        return False

@spaces.GPU
def generate_speech(inp_audio, inp_text, infer_timestep, p_w, t_w):
    if not inp_audio or not inp_text:
        gr.Warning("Please provide both reference audio and text to generate.")
        return None
    
    try:
        print(f"Generating speech with: {inp_text}...")
        
        # Check CUDA availability and clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        else:
            gr.Warning("CUDA is not available. Please check your GPU setup.")
            return None
        
        # Robustly preprocess audio
        try:
            processed_audio_path = preprocess_audio_robust(inp_audio)
            # Use existing cut_wav for final trimming
            cut_wav(processed_audio_path, max_len=28)
            wav_path = processed_audio_path
        except Exception as audio_error:
            gr.Warning(f"Audio preprocessing failed: {str(audio_error)}")
            return None
        
        # Read audio file
        with open(wav_path, 'rb') as file:
            file_content = file.read()
        
        # Generate speech with proper error handling
        try:
            resource_context = infer_pipe.preprocess(file_content)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            # Clean up memory after successful generation
            cleanup_memory()
            return wav_bytes
        except RuntimeError as cuda_error:
            if "CUDA" in str(cuda_error):
                print(f"CUDA error detected: {cuda_error}")
                # Try to reset the model to recover from CUDA errors
                if reset_model():
                    gr.Warning("CUDA error occurred. Model has been reset. Please try again.")
                else:
                    gr.Warning("CUDA error occurred and model reset failed. Please restart the application.")
                return None
            else:
                raise cuda_error
        
    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Speech generation failed: {str(e)}")
        # Clean up CUDA memory on any error
        cleanup_memory()
        return None

def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def preprocess_audio_robust(audio_path, target_sr=22050, max_duration=30):
    """Robustly preprocess audio to prevent CUDA errors."""
    try:
        # Load with pydub for robust format handling
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Limit duration to prevent memory issues
        if len(audio) > max_duration * 1000:  # pydub uses milliseconds
            audio = audio[:max_duration * 1000]
        
        # Normalize audio to prevent clipping
        audio = normalize(audio)
        
        # Convert to target sample rate
        audio = audio.set_frame_rate(target_sr)
        
        # Export to temporary WAV file with specific parameters
        temp_path = audio_path.replace(os.path.splitext(audio_path)[1], '_processed.wav')
        audio.export(
            temp_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr)]
        )
        
        # Validate the audio with librosa
        wav, sr = librosa.load(temp_path, sr=target_sr, mono=True)
        
        # Check for invalid values
        if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
            raise ValueError("Audio contains NaN or infinite values")
        
        # Ensure reasonable amplitude range
        if np.max(np.abs(wav)) < 1e-6:
            raise ValueError("Audio signal is too quiet")
        
        # Re-save the validated audio
        import soundfile as sf
        sf.write(temp_path, wav, sr)
        
        return temp_path
        
    except Exception as e:
        print(f"Audio preprocessing failed: {e}")
        raise ValueError(f"Failed to process audio: {str(e)}")


# Custom CSS for better styling
custom_css = """
.header-container {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    margin-bottom: 20px;
    color: white;
}
.info-box {
    background-color: #f0f4f8;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
}
.warning-box {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin: 10px 0;
}
footer {visibility: hidden}
"""

with gr.Blocks(title="🎙️ DeepNeuralAI Voice Cloning", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # Header Section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-container">
                    <h1 style="margin: 0; font-size: 2.5em;">🎙️ DeepNeuralAI Voice Cloning</h1>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">Advanced AI-Powered Voice Synthesis with MegaTTS3</p>
                </div>
            """)
    
    # About Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 🎯 What is Voice Cloning?
            
            Clone any voice from text using a short reference audio sample. Our advanced AI system captures the speaker's **tone**, **pitch**, 
            and **speaking style** to produce highly personalized voice outputs.
            
            ### 💡 Use Cases
            
            - 🤖 **Personalized Assistants** - Create custom voice assistants
            - 🎨 **Content Creation** - Generate voiceovers for videos and podcasts
            - 📚 **Audiobooks** - Convert written content to spoken audio
            - 📞 **Customer Support** - Automated voice responses
            - ♿ **Accessibility Tools** - Assistive technologies for the disabled
            - 🔬 **Research Applications** - Academic and scientific studies
            """)
    
    # Important Notice
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### ⚠️ Important Ethical Notice
            
            Please use this technology **responsibly**. Voice cloning should only be performed with **proper consent** from the original speaker. 
            This demo is provided strictly for **educational**, **research**, and **authorized use cases**. Misuse of voice cloning technology 
            may violate privacy laws and ethical guidelines.
            """)
    
    gr.Markdown("---")
    
    # Main Interface
    gr.Markdown("### 🚀 Get Started")
    gr.Markdown("Upload a reference audio clip (3-30 seconds recommended) and enter the text you want to generate in the cloned voice.")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("#### 📤 Input Section")
            reference_audio = gr.Audio(
                label="🎵 Reference Audio (3-30 seconds recommended)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            text_input = gr.Textbox(
                label="📝 Text to Generate",
                placeholder="Enter the text you want to synthesize in the cloned voice...",
                lines=5,
                info="Type or paste the text you want to convert to speech"
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.Markdown("*Fine-tune the generation parameters for optimal results*")
                infer_timestep = gr.Slider(
                    label="🔢 Inference Timesteps",
                    value=32,
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="Higher values = better quality but slower (recommended: 32-50)"
                )
                p_w = gr.Slider(
                    label="🎯 Intelligibility Weight",
                    value=1.4,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1,
                    info="Controls clarity and pronunciation (recommended: 1.0-2.0)"
                )
                t_w = gr.Slider(
                    label="🔊 Similarity Weight", 
                    value=3.0,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    info="Controls voice similarity to reference (recommended: 2.0-4.0)"
                )
            
            generate_btn = gr.Button(
                "🎬 Generate Speech", 
                variant="primary", 
                size="lg",
                scale=1
            )
        
        with gr.Column(scale=1):
            gr.Markdown("#### 🎧 Output Section")
            output_audio = gr.Audio(
                label="✨ Generated Audio",
                interactive=False
            )
            
            with gr.Accordion("💡 Tips for Best Results", open=True):
                gr.Markdown("""
                - ✅ Use **clear, high-quality** reference audio
                - ✅ Keep reference audio between **3-30 seconds**
                - ✅ Ensure **minimal background noise**
                - ✅ Use audio with **consistent speaking style**
                - ✅ Test with **different parameter settings**
                - ❌ Avoid extremely short clips (< 2 seconds)
                - ❌ Avoid low-quality or heavily compressed audio
                """)
    
    gr.Markdown("---")
    
    # Examples Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Quick Start Examples")
            gr.Examples(
                examples=[
                    [None, "Hello, welcome to DeepNeuralAI voice cloning. This technology can transform any text into speech using your voice.", 32, 1.4, 3.0],
                    [None, "Artificial intelligence is revolutionizing the way we interact with technology and opening new possibilities every day.", 40, 1.5, 3.5],
                    [None, "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.", 32, 1.4, 3.0],
                ],
                inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
                label="Click to load example text"
            )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; padding: 20px;">
        <p>Powered by <strong>MegaTTS3</strong> | Built with ❤️ by DeepNeuralAI</p>
        <p style="font-size: 0.9em;">For research and educational purposes only</p>
    </div>
    """)
    
    # Event handler
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
        outputs=[output_audio],
        api_name="generate"
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True)
