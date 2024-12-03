# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
ENV COMFYUI_PATH="/comfyui"


# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vmtouch \
    libgl1 \
    libglib2.0-0

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt \
    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager \
    && pip3 install -r custom_nodes/ComfyUI-Manager/requirements.txt

# Install runpod
RUN pip3 install runpod requests

ARG MODEL_TYPE

RUN if [ "$MODEL_TYPE" = "refine" ]; then \
    python3 custom_nodes/ComfyUI-Manager/cm-cli.py install \
        ComfyUI-AdvancedLivePortrait \
        ComfyUI-load-image-from-url \
        ComfyUI-BRIA_AI-RMBG \
    && wget -O custom_nodes/ComfyUI-BRIA_AI-RMBG/RMBG-1.4/model.pth "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth" \
    && pip3 install --upgrade opencv-python \
    && git clone https://github.com/glowcone/comfyui-base64-to-image custom_nodes/comfyui-base64-to-image \
    ; \
  elif [ "$MODEL_TYPE" = "base" ]; then \
    python3 custom_nodes/ComfyUI-Manager/cm-cli.py install \
        ComfyUI_essentials \
        comfyui-various \
        ComfyUI_Comfyroll_CustomNodes \
        was-node-suite-comfyui \
        masquerade-nodes-comfyui \
        ComfyUI-load-image-from-url \
        ComfyUI_UltimateSDUpscale \
        ComfyUI-Inpaint-CropAndStitch \
    && pip3 install --upgrade opencv-python \
    && git clone https://github.com/glowcone/comfyui-base64-to-image custom_nodes/comfyui-base64-to-image \
    ; \
  fi

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG CIVITAI_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "refine" ]; then \
      mkdir -p models/liveportrait models/ultralytics \
        && wget -O models/ultralytics/face_yolov8n.pt "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
        && wget -O models/liveportrait/appearance_feature_extractor.safetensors "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/appearance_feature_extractor.safetensors" \
        && wget -O models/liveportrait/motion_extractor.safetensors "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/motion_extractor.safetensors" \
        && wget -O models/liveportrait/warping_module.safetensors "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/warping_module.safetensors" \
        && wget -O models/liveportrait/spade_generator.safetensors "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/spade_generator.safetensors" \
        && wget -O models/liveportrait/stitching_retargeting_module.safetensors  "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/stitching_retargeting_module.safetensors" \
        ; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start the container
CMD /start.sh
