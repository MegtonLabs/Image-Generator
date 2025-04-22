import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    # Add other schedulers you might want:
    # from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
)
from PIL import Image
import os
import sys
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download # Main download function
from huggingface_hub.utils import HfHubHTTPError # Correct import path for the HTTP error exception
import random
import time
import gc  # Garbage collector
import traceback
from typing import Optional, Tuple, Dict, Any

# --- Configuration Constants ---
APP_TITLE = "Stable Diffusion XL (SDXL) Base 1.0 Generator"
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output")
DEFAULT_MODEL_FILENAME = "sdxl_base_1.0.safetensors"
MODEL_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Highly recommended for SDXL quality, especially with fp16
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
# VAE_MODEL_ID = None

# Set vae_model_id = None to disable loading the external VAE and use the one baked into the main model

# --- Global State Variables ---
# Initialized during startup, holding loaded models/config
pipeline: Optional[StableDiffusionXLPipeline] = None
current_device: str = "cpu"  # Will be updated
current_dtype: torch.dtype = torch.float32  # Will be updated
available_schedulers: Dict[str, Any] = {}  # Populated after pipeline load

# --- Utility Functions ---

def cleanup_memory():
    """Attempts to clear CUDA cache and run garbage collection."""
    print("Running memory cleanup...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   CUDA cache cleared.")
    gc.collect()
    print("   Garbage collection finished.")

def select_device() -> Tuple[str, torch.dtype]:
    """Detects and selects the best available device (GPU preferred)."""
    print("1. Selecting Device...")
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # Optimal VRAM usage on CUDA
        print("   Found CUDA (NVIDIA GPU). Using 'cuda' with float16.")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16  # MPS also benefits from float16
        print("   Found MPS (Apple Silicon GPU). Using 'mps' with float16.")
    else:
        print("\n" + "=" * 30)
        print(" ERROR: No CUDA or MPS GPU detected! ")
        print("=" * 30)
        print(" This application requires an NVIDIA GPU (CUDA) or Apple Silicon GPU (MPS).")
        print(" CPU execution is not supported.")
        sys.exit(1)  # Exit if no compatible GPU found
    return device, dtype

def download_model_if_needed(model_path: Path, repo_id: str, filename: str):
    """Downloads the specified model from Hugging Face Hub if it doesn't exist locally."""
    print(f"2. Checking for model file: {model_path}")
    if not model_path.exists():
        print(f"   Model not found. Downloading '{filename}' from '{repo_id}'...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_path.parent,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"   Model downloaded successfully to '{model_path}'")
        # Correctly catches the exception now that it's imported properly
        except HfHubHTTPError as e:
             print("\n" + "="*30)
             print(f" ERROR: Failed to download model (HTTP Error)! ")
             print(f"   Status Code: {e.response.status_code}") # Access response if needed
             print(f"   Error details: {e}")
             print("="*30)
             print(f" Please ensure internet connectivity and that the repository/file exists:")
             print(f"   Repo: '{repo_id}', File: '{filename}'")
             sys.exit(1)
        except Exception as e:
            print("\n" + "=" * 30)
            print(f" ERROR: Failed to download model (Other Error)! ")
            print(f"   Error details: {e}")
            print("=" * 30)
            print(f" You can also manually download '{filename}' from {repo_id} ")
            print(f" and place it in the '{model_path.parent}' directory.")
            sys.exit(1)
    else:
        print(f"   Model found locally.")

def load_vae_model(device: str, dtype: torch.dtype) -> Optional[AutoencoderKL]:
    """Loads the VAE model if VAE_MODEL_ID is specified."""
    if not VAE_MODEL_ID:
        print("   Skipping external VAE loading as VAE_MODEL_ID is not set.")
        return None

    print(f"   Loading VAE: {VAE_MODEL_ID}...")
    try:
        vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_ID,
            torch_dtype=dtype  # Load VAE in the target precision
        )
        vae.to(device)  # Move VAE to the selected device
        print(f"   VAE loaded and moved to '{device}'.")
        return vae
    except Exception as e:
        print(f"   Warning: Could not load VAE {VAE_MODEL_ID}. Error: {e}")
        print("   Proceeding without external VAE (will attempt to use baked-in).")
        return None

def load_pipeline_model(model_path: Path, vae: Optional[AutoencoderKL], device: str, dtype: torch.dtype):
    """Loads the main Stable Diffusion XL pipeline."""
    global pipeline, available_schedulers  # Allow modification of global state

    print(f"   Loading SDXL Pipeline from: {model_path}...")
    pipeline_load_start_time = time.time()
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            vae=vae,  # Pass the pre-loaded VAE (or None)
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None, # Specify variant if using fp16
            use_safetensors=True,
            # Add other components if needed, e.g., safety checker (often disabled for local use)
            # safety_checker=None, requires_safety_checker=False,
        )
        print(f"   SDXL Pipeline created (took {time.time() - pipeline_load_start_time:.2f}s).")

        # Populate available schedulers based on the loaded pipeline's config
        scheduler_config = pipe.scheduler.config
        available_schedulers = {
            "Euler": EulerDiscreteScheduler.from_config(scheduler_config),
            "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
            "UniPC": UniPCMultistepScheduler.from_config(scheduler_config),
            # Add more scheduler instances here if desired:
            # "DDIM": DDIMScheduler.from_config(scheduler_config),
            # "LMS": LMSDiscreteScheduler.from_config(scheduler_config),
            # "PNDM": PNDMScheduler.from_config(scheduler_config),
        }
        print(f"   Available schedulers: {', '.join(available_schedulers.keys())}")

        # Move the entire pipeline to the selected device
        print(f"   Moving pipeline to '{device}'...")
        pipeline_move_start_time = time.time()
        pipe.to(device)
        print(f"   Pipeline successfully moved to '{device}' (took {time.time() - pipeline_move_start_time:.2f}s).")
        pipeline = pipe  # Assign to global variable *only after* successful loading and moving

    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 30)
        print(" ERROR: CUDA Out of Memory! (During Pipeline Loading/Moving) ")
        print("=" * 30)
        print(" Your GPU likely does not have enough VRAM to load the SDXL model (~10-12GB+ recommended).")
        print(" Try closing other applications using the GPU.")
        cleanup_memory()
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 30)
        print(f" ERROR: Failed to load pipeline! ")
        print(f"   Error details: {e}")
        print("=" * 30)
        print(" Check model file integrity, dependencies (diffusers, transformers, torch, etc.),")
        print(" and ensure your GPU driver is up-to-date.")
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

def initialize_models():
    """Sequentially sets up the device, downloads, and loads models."""
    global current_device, current_dtype

    # Ensure directories exist
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_start_time = time.time()
    current_device, current_dtype = select_device() # Sets global vars
    model_file_path = CHECKPOINTS_DIR / DEFAULT_MODEL_FILENAME # Define path here
    download_model_if_needed(model_file_path, MODEL_REPO_ID, DEFAULT_MODEL_FILENAME)

    print("3. Loading Models...")
    vae = load_vae_model(current_device, current_dtype)
    load_pipeline_model(model_file_path, vae, current_device, current_dtype) # Use defined path

    if pipeline is None:
        print("\nFATAL: Pipeline initialization failed. Exiting.")
        sys.exit(1)

    print(f"\nModel loading complete. Total setup time: {time.time() - setup_start_time:.2f}s")
    cleanup_memory() # Run GC after large model loads


# --- Gradio UI Related Functions ---

def apply_vram_optimizations(vram_mode: str):
    """Configures pipeline optimizations based on selected VRAM mode."""
    global pipeline # Ensure we're modifying the global pipeline object
    if not pipeline:
        print("Warning: Pipeline not available for VRAM optimization.")
        return

    print(f"\nApplying VRAM Optimisation: {vram_mode}")
    # --- Reset relevant states ---
    # VAE slicing can be explicitly disabled.
    pipeline.disable_vae_slicing()
    print("   - VAE Slicing Disabled (Reset).")

    # --- Apply specific optimizations ---
    # Note: There's no explicit 'disable_sequential_cpu_offload'.
    # The default state (after .to(device)) is *not* offloaded.
    # We only *enable* it when needed for low VRAM.

    if vram_mode == "Low VRAM (<= 8GB)":
        # Strongest memory saving, potentially slower inference
        pipeline.enable_vae_slicing()
        print("   + VAE Slicing Enabled.")
        # Enable CPU offload ONLY for this mode
        pipeline.enable_sequential_cpu_offload()
        print("   + Sequential CPU Offload Enabled.")

    elif vram_mode == "Medium VRAM (8-12GB)":
        # Good balance of memory saving and speed
        pipeline.enable_vae_slicing()
        print("   + VAE Slicing Enabled.")
        # Ensure CPU offload is *not* enabled (it shouldn't be unless previously set,
        # but being explicit doesn't hurt, though there's no disable function).
        # The lack of calling .enable_sequential_cpu_offload() ensures it's off.
        print("   - Sequential CPU Offload remains Disabled.")

    elif vram_mode == "High VRAM (>12GB)":
        # Fastest inference, highest memory usage
        # Both VAE slicing and CPU offload remain disabled (their reset state).
        print("   - VAE Slicing remains Disabled.")
        print("   - Sequential CPU Offload remains Disabled (Max performance).")

    # Optional: Handle XFormers (often automatic if installed)
    # try:
    #     pipeline.enable_xformers_memory_efficient_attention()
    #     print("   + XFormers Enabled (if available).")
    # except Exception:
    #     print("   - XFormers not available or failed to enable.")
    #     pass # Ignore if xformers not available/fails


def generate_image(
    prompt: str,
    negative_prompt: str,
    vram_optimisation: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler_name: str,
    seed_num: int,
    use_random_seed: bool,
    progress=gr.Progress(track_tqdm=True) # Gradio progress tracker
) -> Tuple[Optional[Image.Image], str]:
    """Callback function for Gradio button click. Generates the image."""
    global pipeline, current_device

    if pipeline is None:
        gr.Error("Pipeline not loaded. Please check the console for errors during startup.")
        return None, "Error: Pipeline not available"

    # --- 1. Input Validation & Preparation ---
    if not prompt:
        gr.Warning("Prompt cannot be empty!")
        return None, "Status: Error - Prompt is empty"

    # Ensure dimensions are multiples of 8 (required by SDXL VAE) and within limits
    original_width, original_height = width, height
    width = max(64, min(1024, (width // 8) * 8))
    height = max(64, min(1024, (height // 8) * 8))
    if width != original_width or height != original_height: # Check if clamping occurred
         gr.Warning(f"Dimensions adjusted to be multiples of 8 and within 64-1024. New size: {width}x{height}")
         print(f"   Dimensions adjusted from {original_width}x{original_height} to {width}x{height}.")


    # Determine seed
    if use_random_seed or seed_num is None or seed_num < 0:
        actual_seed = random.randint(0, 2**32 - 1)
    else:
        actual_seed = int(seed_num)
    generator = torch.Generator(device=current_device).manual_seed(actual_seed)

    print(f"\n--- Starting Generation ---")
    print(f"  Seed: {actual_seed}{' (Random)' if use_random_seed or seed_num < 0 else ''}")

    # --- 2. Configure Pipeline for this run ---
    apply_vram_optimizations(vram_optimisation) # Call the corrected function

    # Select and set the scheduler
    if scheduler_name in available_schedulers:
        pipeline.scheduler = available_schedulers[scheduler_name]
        print(f"  Using Scheduler: {scheduler_name}")
    else:
        default_scheduler_name = pipeline.scheduler.__class__.__name__
        gr.Warning(f"Scheduler '{scheduler_name}' not found. Using default: {default_scheduler_name}")
        print(f"  Warning: Scheduler '{scheduler_name}' not found. Using default: {default_scheduler_name}")
        # Pipeline already has its default scheduler loaded, so no change needed here

    # --- 3. Run Inference ---
    status = f"Status: Generating with seed: {actual_seed}..."
    print(f"  Prompt: '{prompt[:100]}{'...' if len(prompt)>100 else ''}'") # Log truncated prompt
    if negative_prompt:
        print(f"  Negative Prompt: '{negative_prompt[:100]}{'...' if len(negative_prompt)>100 else ''}'")
    print(f"  Settings: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")

    output_image = None
    generation_start_time = time.time()
    try:
        # --- CORRECTED CALLBACK DEFINITION ---
        # Define progress callback for Gradio UI
        # It must accept the pipeline instance, step, timestep, and kwargs dict
        def progress_callback(pipe_instance, step: int, timestep: torch.Tensor, callback_kwargs: Dict[str, Any]):
            """Updates the Gradio progress bar."""
            # We only need the 'step' for the progress bar calculation
            current_steps = int(num_inference_steps) # Ensure it's an int for comparison
            if current_steps > 0:
                 # step starts from 0, step+1 is the current step number
                 progress((step + 1) / current_steps, desc=f"Sampling Step {step + 1}/{current_steps}")
            else:
                 progress(0, desc="Sampling Step ?/?")
            # It's standard practice for callbacks modifying kwargs to return them
            return callback_kwargs
        # --- END OF CORRECTION ---

        with torch.inference_mode():  # Ensures no gradients are calculated
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps), # Ensure integer
                generator=generator,
                # Pass the correctly defined callback function
                callback_on_step_end=progress_callback if int(num_inference_steps) > 0 else None, # Newer diffusers
                # Specify which tensors the callback should receive in its kwargs dictionary
                callback_on_step_end_tensor_inputs=["latents"] # We ask for latents, even if not used in this specific progress func
            )
            output_image = result.images[0]

        generation_time = time.time() - generation_start_time
        print(f"  Image generation successful (took {generation_time:.2f}s).")
        status = f"Status: Success! Seed: {actual_seed}"

        # --- 4. Save Image ---
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = OUTPUT_DIR / f"sdxl_{timestamp}_seed{actual_seed}.png"
            output_image.save(output_filename)
            print(f"  Image saved to: {output_filename}")
            status += f" | Saved: {output_filename.name}"
        except Exception as save_e:
            print(f"  Error saving image: {save_e}")
            gr.Warning(f"Image generated but failed to save: {save_e}")
            status += " | Save Failed"


    except torch.cuda.OutOfMemoryError:
        generation_time = time.time() - generation_start_time
        print(f"\nERROR: CUDA Out of Memory during image generation! (Attempted for {generation_time:.2f}s)")
        error_msg = ("OOM Error: Not enough VRAM for the current settings. Try:\n"
                     "• Reducing Width/Height.\n"
                     "• Selecting 'Low VRAM' optimisation.\n"
                     "• Closing other GPU-intensive applications.")
        print(error_msg.replace("• ", "  - ")) # Console friendly list
        gr.Error(error_msg)
        status = "Status: Error - CUDA Out of Memory"
        cleanup_memory() # Try to free memory after OOM

    except Exception as e:
        generation_time = time.time() - generation_start_time
        print(f"\nERROR: An unexpected error occurred during generation: {e} (Occurred after {generation_time:.2f}s)")
        traceback.print_exc() # Print detailed traceback to console
        gr.Error(f"An unexpected error occurred: {e}")
        status = f"Status: Error - {e}"
        cleanup_memory() # Try to free memory after other errors too

    print("--- Generation Finished ---")
    # Return the PIL image and the status string
    return output_image, status

# --- Build Gradio Interface ---

def create_gradio_ui() -> gr.Blocks:
    """Defines and returns the Gradio interface layout."""
    print("4. Creating Gradio Interface...")
    with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown("Generate images using the SDXL 1.0 base model with various controls.")

        with gr.Row():
            # --- Left Column: Inputs ---
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3,
                    elem_id="prompt_input"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter things to avoid in the image...",
                    lines=2,
                    value="ugly, deformed, blurry, low quality, text, watermark, signature, noise, frame, border", # Sensible default
                    elem_id="neg_prompt_input"
                )

                vram_optimisation = gr.Radio(
                    label="VRAM Usage Mode",
                    choices=["Low VRAM (<= 8GB)", "Medium VRAM (8-12GB)", "High VRAM (>12GB)"],
                    value="Medium VRAM (8-12GB)",
                    info="Adjust based on your GPU VRAM for performance/stability.",
                    elem_id="vram_mode_radio"
                )

                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=1024, elem_id="width_slider")
                    height = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=1024, elem_id="height_slider")

                with gr.Row():
                    guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, step=0.5, value=7.0, info="How strictly to follow the prompt (higher = stricter).", elem_id="cfg_slider")
                    num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, step=1, value=25, info="Number of steps in the generation process.", elem_id="steps_slider")

                scheduler_name = gr.Dropdown(
                    label="Scheduler/Sampler",
                    choices=list(available_schedulers.keys()), # Use dynamically populated list
                    value="DPM++ 2M Karras", # A common, good default
                    info="Algorithm for the diffusion denoising process.",
                    elem_id="scheduler_dropdown"
                )

                with gr.Row(equal_height=True):
                    seed_num = gr.Number(label="Seed", value=-1, precision=0, elem_id="seed_number", scale=3)
                    use_random_seed = gr.Checkbox(label="Use Random Seed", value=True, elem_id="random_seed_checkbox", scale=1, min_width=150) # Adjust scale/min_width


                generate_btn = gr.Button("Generate Image", variant="primary", elem_id="generate_button")

            # --- Right Column: Outputs ---
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    interactive=False,
                    elem_id="output_image",
                    height=512 # Set a default display height
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="Status: Ready",
                    interactive=False,
                    lines=3, # Allow more lines for error messages
                    elem_id="status_textbox"
                 )

        # --- Footer / Info Area ---
        gr.Markdown("---")
        model_name = DEFAULT_MODEL_FILENAME
        vae_name = VAE_MODEL_ID.split('/')[-1] if VAE_MODEL_ID else "Baked-in"
        dtype_name = str(current_dtype).split('.')[-1]
        gr.Markdown(f"**Model:** `{model_name}` | **VAE:** `{vae_name}` | **Device:** `{current_device.upper()}` | **Precision:** `{dtype_name}`")
        gr.Markdown(f"**Output Directory:** `{OUTPUT_DIR.resolve()}`")


        # --- Event Handling ---
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, vram_optimisation, width, height,
                guidance_scale, num_inference_steps, scheduler_name,
                seed_num, use_random_seed,
            ],
            outputs=[output_image, status_text]
        )

        # Link checkbox to interactivity of the seed number input
        def toggle_seed_interactive(is_random):
            return gr.update(interactive=not is_random)

        use_random_seed.change(
            fn=toggle_seed_interactive,
            inputs=use_random_seed,
            outputs=seed_num
        )

        # Set initial state of seed interactivity on UI load
        demo.load(lambda is_random: gr.update(interactive=not is_random), inputs=use_random_seed, outputs=seed_num)


    print("   Gradio UI defined.")
    return demo

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Starting {APP_TITLE}...")
    try:
        # --- 1. Initialize Backend ---
        initialize_models() # Sets up device, downloads, loads VAE & Pipeline

        # --- 2. Create and Launch Frontend ---
        gradio_ui = create_gradio_ui()
        print("\n5. Launching Gradio Interface...")
        print(f"   Access it locally at the URL provided by Gradio (usually http://127.0.0.1:7860)")
        gradio_ui.launch() # share=True for public link (use with caution)

    except SystemExit:
        print("\nExiting application due to setup error or user request.")
    except KeyboardInterrupt:
         print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred during startup or runtime: {e}")
        traceback.print_exc()
    finally:
        print("\nApplication shutdown.")
        cleanup_memory() # Final cleanup attempt