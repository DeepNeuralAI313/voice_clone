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
    
    # Check if critical config files exist
    critical_files = [
        "checkpoints/duration_lm/config.yaml",
        "checkpoints/aligner_lm/config.yaml",
        "checkpoints/diffusion_transformer/config.yaml",
        "checkpoints/wavvae/config.yaml"
    ]
    
    needs_download = not os.path.exists(weights_dir) or not all(os.path.exists(f) for f in critical_files)
    
    if needs_download:
        print("Downloading model weights from HuggingFace...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=weights_dir,
            local_dir_use_symlinks=False
        )
        print("Model weights downloaded successfully!")
        
        # Verify download
        missing_files = [f for f in critical_files if not os.path.exists(f)]
        if missing_files:
            print(f"Warning: Missing critical files after download: {missing_files}")
            print(f"Contents of checkpoints directory:")
            if os.path.exists(weights_dir):
                for root, dirs, files in os.walk(weights_dir):
                    for file in files[:20]:  # Show first 20 files
                        print(f"  {os.path.join(root, file)}")
    else:
        print("Model weights already exist.")
    
    return weights_dir


# Download weights and initialize model
weights_dir = download_weights()

# Verify critical paths exist before initializing model
print(f"Working directory: {os.getcwd()}")
print(f"Checkpoints exist: {os.path.exists('checkpoints')}")
if os.path.exists('checkpoints'):
    print(f"Checkpoints contents: {os.listdir('checkpoints')}")

print("Initializing MegaTTS3 model...")
try:
    infer_pipe = Voice_cloning_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure all checkpoint files are properly downloaded.")
    raise

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


with gr.Blocks(title="DeepNeuralAI Voice Cloning", theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.Markdown("# DeepNeuralAI Voice Cloning")
    

    with gr.Row():
        with gr.Column():
            reference_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to synthesize...",
                lines=3
            )
            
            with gr.Accordion("Advanced Options", open=False):
                infer_timestep = gr.Number(
                    label="Inference Timesteps",
                    value=32,
                    minimum=1,
                    maximum=100,
                    step=1
                )
                p_w = gr.Number(
                    label="Intelligibility Weight",
                    value=1.4,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1
                )
                t_w = gr.Number(
                    label="Similarity Weight", 
                    value=3.0,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
        outputs=[output_audio]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True)
