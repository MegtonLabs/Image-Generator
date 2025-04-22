import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import os
import sys # Import sys to exit the script
from pathlib import Path

# --- Configuration ---
model_path = os.path.join("checkpoints", "sdxl_base_1.0.safetensors")

# It's often better to use a dedicated SDXL VAE for potentially better results
vae_model_id = "madebyollin/sdxl-vae-fp16-fix"
# Use None if you want to try using the VAE potentially baked into the base model file
# vae_model_id = None

output_filename = "generated_image_sdxl_gpu.png"
prompt = "An astronaut riding a majestic horse on the moon, detailed painting, 4k"
negative_prompt = "ugly, deformed, blurry, low quality, text, watermark, signature"

# --- Setup Device (GPU ONLY) ---
print("Checking for compatible GPU...")
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16 # Use float16 for GPU memory saving
    print("CUDA (NVIDIA GPU) detected.")
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16 # MPS also benefits from float16
    print("MPS (Apple Silicon GPU) detected.")
else:
    print("\n" + "="*30)
    print(" ERROR: No compatible GPU detected! ")
    print("="*30)
    print("This script requires either an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS).")
    print("CPU execution is not supported by this modified script.")
    sys.exit(1) # Exit the script with an error code

print(f"Using device: {device}")
print(f"Using dtype: {torch_dtype}")

# --- Load VAE (Recommended) ---
vae = None
if vae_model_id:
    try:
        print(f"\nLoading VAE: {vae_model_id}...")
        vae = AutoencoderKL.from_pretrained(
            vae_model_id,
            torch_dtype=torch_dtype # Load VAE in the target precision
        )
        print("VAE loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load VAE {vae_model_id}. Error: {e}")
        print("Proceeding without external VAE (will try to use baked-in or default).")
        vae = None # Ensure vae is None if loading failed

# --- Load Pipeline ---
print(f"\nLoading SDXL Base model from: {model_path}...")
# If a specific VAE is loaded, pass it. Otherwise, let the pipeline handle it.
# `from_single_file` will load the UNet and Text Encoders from the safetensors file.
# If vae=None, it will try to load one from the file or use a default if necessary.
try:
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        vae=vae, # Pass the loaded VAE here, or None
        torch_dtype=torch_dtype,
        variant="fp16", # Force fp16 variant since we require GPU
        use_safetensors=True # Explicitly state we are using safetensors
    )
    print("SDXL Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("Check file path, dependencies, model integrity, and GPU compatibility.")
    sys.exit(1) # Exit if pipeline loading fails

# --- Move to Device ---
print(f"\nMoving pipeline to {device}...")
try:
    pipe.to(device)
    print("Pipeline moved to device.")
    # Optional: Enable memory-efficient attention if available/needed (requires xformers)
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    #     print("Enabled xformers memory efficient attention.")
    # except Exception:
    #     print("xformers not available or failed to enable.")
    # Optional: Sliced VAE decode/encode for lower VRAM usage (at slight speed cost)
    # pipe.enable_vae_slicing()

except torch.cuda.OutOfMemoryError:
     print("\nCUDA Out of Memory! Moving pipeline failed.")
     print("Ensure your GPU has enough VRAM for SDXL (ideally 12GB+).")
     print("Try closing other GPU-intensive applications.")
     sys.exit(1)
except Exception as e:
     print(f"Error moving pipeline to device: {e}")
     sys.exit(1)

# --- Generate Image ---
print(f"\nGenerating image with prompt: '{prompt}'...")
try:
    with torch.no_grad(): # Ensure gradients aren't computed
        image: Image.Image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=30,
            # width=1024, # Default SDXL width
            # height=1024, # Default SDXL height
        ).images[0] # The output is a list of images, get the first one
    print("Image generated successfully.")

    # --- Save Image ---
    image.save(output_filename)
    print(f"Image saved as: {output_filename}")

except torch.cuda.OutOfMemoryError:
    print("\nCUDA Out of Memory during image generation!")
    print("Your GPU may not have enough VRAM for the default SDXL resolution (1024x1024) even at fp16.")
    print("Try closing other applications, enabling VAE slicing (`pipe.enable_vae_slicing()`),")
    print("or uncommenting `pipe.enable_xformers_memory_efficient_attention()` if xformers is installed.")
    print("Consider using a smaller model or lower resolution if supported/possible.")
except Exception as e:
    print(f"\nAn error occurred during image generation: {e}")

print("\nScript finished.")