import os
import re # For robust JSON cleaning
# Optimization for low VRAM and Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import gc
import random
import time
import glob
import numpy as np
import imageio
import cv2
from PIL import Image
from wakepy import keep
from datetime import datetime
import shutil

# --- Constants & Config ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models
# Switched to SDXL-Turbo (Native, faster, fixes config crash)
MODEL_SDXL = "stabilityai/sdxl-turbo" 
MODEL_SVD_XT = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
MODEL_MOONDREAM = "vikhyatk/moondream2" 
HF_TOKEN = ""
OLLAMA_URL = "http://localhost:11434/api/generate"
# Updated to user preference
OLLAMA_MODEL = "gemma3:4b" 

# Global State
current_model = None
loaded_dream_models = {} 
stop_dreaming_flag = False # Flag to interrupt the loop
pending_guide_question = ""
guide_disabled_until = 0

# --- Helpers ---
def clean_json_text(text):
    """Robustly cleans LLM output to extract just the JSON."""
    # Remove markdown code blocks
    if "```" in text:
        # pattern to find content between ```json and ``` or just ``` and ```
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # If no markdown, try to find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    
    return text.strip()

# --- Memory Management ---

def cleanup_memory():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def unload_model(model):
    """Unloads a specific model from GPU memory."""
    if model is not None:
        del model
    cleanup_memory()

def unload_dream_stack_only():
    """Aggressively clears VAE, RNN, and Guide models for expansion."""
    global loaded_dream_models
    
    # 1. Unload VAE/RNN
    unload_model(loaded_dream_models.pop("vae", None))
    unload_model(loaded_dream_models.pop("rnn", None))
    # 2. Unload Guide
    guide_tuple = loaded_dream_models.pop("guide", None)
    if guide_tuple:
        unload_model(guide_tuple[0]) # model
        # Tokenizer is usually lightweight, but we track the disposal
    
    loaded_dream_models.clear()
    cleanup_memory()
    print("🧹 Successfully cleared VAE/RNN/Guide stack for Builder.")

def unload_heavy_models():
    """Aggressively clears VRAM of the global current_model."""
    global current_model
    if current_model is not None:
        print(f"🧹 Unloading {type(current_model).__name__}...")
        unload_model(current_model)
        current_model = None
    
    # Also clear Dream models if we are building
    if loaded_dream_models:
        print("🧹 Clearing Dream Stack...")
        loaded_dream_models.clear()
    
    cleanup_memory()

# --- Neural Networks (The Brains) ---
class SimpleVAE(nn.Module):
    """The Eye: Compresses Reality (64x64) into Latents (32 dim)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, 32)
        self.fc_logvar = nn.Linear(128*8*8, 32)
        self.dec_fc = nn.Linear(32, 128*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.decoder(h_dec), mu, logvar

class SimpleRNN(nn.Module):
    """The Traveler: Predicts the next latent vector."""
    def __init__(self):
        super().__init__()
        # INCREASED HIDDEN SIZE: 128 -> 256 for better motion modeling
        self.rnn = nn.LSTM(32 + 3, 256, batch_first=True) 
        self.fc = nn.Linear(256, 32) # Must match the new hidden size

    def forward(self, z, action, hidden=None):
        input_combined = torch.cat([z, action], dim=1).unsqueeze(1)
        out, hidden = self.rnn(input_combined, hidden)
        pred_z = self.fc(out.squeeze(1))
        return pred_z, hidden

# --- Tab 1: Build (The Cartography) ---

# --- History / Diversity Utils ---
def get_recent_history(limit=10):
    """Reads the last N entries from history.jsonl."""
    history_file = os.path.join(PROJECT_ROOT, "history.jsonl")
    if not os.path.exists(history_file):
        return []
    
    lines = []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        recent_descs = []
        for line in lines[-limit:]:
            try:
                entry = json.loads(line)
                if "description" in entry:
                    recent_descs.append(entry["description"])
            except:
                continue
        return recent_descs
    except Exception as e:
        print(f"History Read Error: {e}")
        return []

def append_to_history(description):
    """Appends a new generation to history.jsonl."""
    history_file = os.path.join(PROJECT_ROOT, "history.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "description": description
    }
    try:
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"History Write Error: {e}")

def generate_trip_concept(vibe, custom_prompt=None):
    """Generates a rich location description."""
    print(f"Trip Planner: Planning a '{vibe}' trip...")
    
    # Logic to handle "Wander Further" context (passed via custom_prompt usually)
    prompt_context = ""
    if custom_prompt and "CONTEXT:" in custom_prompt:
        prompt_context = custom_prompt
        # Reset custom prompt so it doesn't get stuck
        custom_prompt = None


    # 1. --- HIGH PRIORITY: CUSTOM PROMPT ---
    if custom_prompt and custom_prompt.strip():
        prompt = f"""You are a creative travel agent describing a location to a potential traveler when they said they wanted: {custom_prompt}. Explain the location in one brief sentence (e.g. "It's a vast desert.","It's a metropolis."), then provide a vivid and atmospheric description of this specific place that a traveler would like to explore. Focus on visual details. Keep the description of the location under 50 words and respond ONLY with that description, no other explanation."""
    
    # 2. --- SECOND PRIORITY: AUTONOMOUS/WANDER CONTEXT ---
    elif vibe == "Autonomous":
        # --- DIVERSITY ENFORCEMENT ---
        recent_history = get_recent_history(10)
        diversity_instruction = ""
        if recent_history:
            history_text = "\n".join([f"- {desc[:100]}..." for desc in recent_history])
            diversity_instruction = f"""
            CRITICAL INSTRUCTION: You must suggest a location for the traveler that is SUBSTANTIVELY DIFFERENT from these recent locations:
            {history_text}
            Do not repeat themes, biomes, color palettes, or atmospheres found in this list of locations you've suggested before. Be unique.
            """
            print(f"Trip Planner: Enforcing diversity against {len(recent_history)} recent items.")

        if prompt_context:
            # Use context gathered from Wander Further Prep
            prompt = f"""You are a creative travel agent describing a new location to a traveler. The traveler wants to leave their current location for a NEW area. Write a vivid and atmospheric description of the NEW area. You MUST begin by explaining the location in one brief sentence (e.g. "It's a vast desert.","It's a metropolis.","It's a beautiful beach.","It's a cruise ship."), then provide a vivid and atmospheric description of this location that a traveler would like to explore. Focus on visual details. Keep the description of the location under 50 words and respond ONLY with that description, no other explanation. 
            
            CURRENT LOCATION: {prompt_context}
            {diversity_instruction}"""
        else:
            # Default Autonomous prompt
            prompt = f"""You are a creative travel agent describing a location to a potential traveler. Everyone loves to visit a relaxing beach, or perhaps a fantasy forest, or a futuristic city, or even visit onboard a cruise ship. You MUST begin by explaining the location in one brief sentence (e.g. "It's a vast desert.","It's a metropolis.","It's a beautiful beach.","It's a cruise ship."), then provide a vivid and atmospheric description of this location that a traveler would like to explore. Focus on visual details. Keep the description of the location under 50 words and respond ONLY with that description, no other explanation.
            {diversity_instruction}"""
    
    # 3. --- LOWEST PRIORITY: VIBE DROPDOWN ---
    else:
        prompt = f"""You are a creative travel agent describing a location to a potential traveler when they said they wanted this vibe: {vibe}. You MUST begin by explaining the location in one brief sentence (e.g. "It's a vast desert.","It's a metropolis.","It's a beautiful beach.","It's a cruise ship."), then provide a vivid and atmospheric description of this specific vibe that a traveler would like to explore. Focus on visual details. Keep the description of the location under 50 words and respond ONLY with that description, no other explanation."""
    

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"num_gpu": 0, "temperature": 0.9}
        })
        response.raise_for_status()
        location_desc = response.json()['response'].strip()
        print(f"Trip Planner: Location found - {location_desc[:50]}...")
        
        # --- LOGGING ---
        if vibe == "Autonomous":
            append_to_history(location_desc)
            
        return location_desc
    except Exception as e:
        print(f"Trip Planner Error: {e}")
        return "A mysterious, foggy void with faint neon lights in the distance."

def generate_shot_list(location_desc):
    """Generates a JSON shot list with SPATIAL logic."""
    print("Travel Agent: Generating shot list...")
    
    # 1. THE NEW PROMPT (Compass Logic + Context Anchoring)
    system_prompt = """You are a technical director for a virtual world.
    Your task is to generate a strict JSON shot list based on a location description. You must use the exact keys provided and you must provide a visual shot description for each key.

    ### CRITICAL RULES for describing every shot:
    1.  **Spatial Logic:** Imagine standing in the center.
        * **hub**: The 360-degree establishing shot.
        * **forward**: North. **back**: South. **left**: West. **right**: East.
    2.  **Lighting Consistency:** If Sun is Forward, Back must be backlit/bright.
    3.  **Landmark Logic:** If a mountain is Forward, it CANNOT be in the Back shot.
    4.  **LENGTH CONSTRAINT:** Each of your shot descriptions must be 50 words or less. Since the hub forms the basis of a further description for at least the forward shot, the hub description must be less than 31 words.
    5.  **CONTEXT RETENTION (MOST IMPORTANT):** * You MUST repeat the location name/theme in EVERY description.
        * **IMPORTANT:** The "forward" and "back" shots are EXPLORATION shots. They must describe the visual scene and then "Camera moves forward".
        * **Formula for forward:** "forward": "[Copy Hub Description]. Camera moves forward..."
        * **Formula for back:** "back": "[Description of a different view facing South]. Camera moves forward..." (Do not indicate turning. We're faced in the opposite direction and are moving forward into the area behind us.)
        * **Formula for left/right:** Use specific panning language: "Camera pans from center to left" or "Camera pans from center to right", and describe the visuals that are revealed in the scene.

    ### OUTPUT FORMAT:
    Return ONLY a raw JSON object. Keys: "hub", "forward", "left", "right", "back".
    
    In the below example:
    - Note the shorter description for hub, which is then copied to forward, which then adds camera movement. 
    - Other shots DON'T copy/paste the hub description, but beautifully describe the scene.
    - Hub description is under 30 words and all descriptions are under 50 words. 
    - Left and Right include direction for clear camera motion.
    - Forward and Back indicate direction for camera moving forward, but they never mention turning or panning.
    {
       "hub": "Golden hour in a vast Desert Canyon. Towering red cliffs frame a sandy floor. The low sun casts long, dramatic shadows.",
       "forward": "Golden hour in a vast Desert Canyon. Towering red cliffs frame a sandy floor. The low sun casts long, dramatic shadows. Camera moves forward through the Canyon directly toward the blinding sun.",
       "back": "Facing south in a vast Desert Canyon, away from the sun. The rock walls are fully illuminated in vibrant orange and red light against a deep blue sky. Camera moves forward exploring the illuminated path.",
       "left": "Panning West in a vast Desert Canyon. The cliff wall is side-lit, revealing the deep, rough texture of the red rock. Camera pans from center to left, revealing the cactus shadows.",
       "right": "Panning East in a vast Desert Canyon. The canyon widens here. The uneven terrain is highlighted by grazing light. Camera pans from center to right, showing the widening path."
    }
    """
    
    full_prompt = f"{system_prompt}\n\nLOCATION: {location_desc}\n\nNow return your visually described shot list in JSON format."
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False,
            "options": {"num_gpu": 0, "temperature": 0.7, "format": "json"}
        })
        response.raise_for_status()
        raw_text = response.json()['response'].strip()
        cleaned_text = clean_json_text(raw_text)
        return json.loads(cleaned_text)
        
    except Exception as e:
        print(f"Travel Agent Error: {e}")
        return {
            "hub": f"Wide shot of {location_desc}",
            "forward": f"Moving forward in {location_desc}",
            "left": f"Looking left in {location_desc}",
            "right": f"Looking right in {location_desc}",
            "back": f"Looking back in {location_desc}"
        }

# --- Media Utils ---
def save_video_frames(frames, output_path, fps=24):
    try:
        np_frames = [np.array(frame) for frame in frames]
        imageio.mimwrite(output_path, np_frames, fps=fps, codec="libx264", quality=9)
        print(f"Saved video to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save video frames: {e}")

# --- Part 1: Images Only (SDXL) ---
def generate_images(shot_list, world_name="my_world", progress=gr.Progress()):
    """Phase 1: Generates the 5 static images using SDXL."""
    global current_model
    timestamp = int(time.time())
    job_dir = os.path.join(OUTPUT_DIR, world_name, f"room_{timestamp}")
    os.makedirs(job_dir, exist_ok=True)
    
    # Empty state for video placeholders
    yield None, None, None, None, None, "🎨 Loading Image Generator...", job_dir
    
    # 1. Load SDXL
    if current_model is None or not hasattr(current_model, "text_encoder") or "LTX" in str(type(current_model)):
        unload_model(current_model)
        try:
            print("Loading SDXL-Turbo...")
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe = AutoPipelineForText2Image.from_pretrained(MODEL_SDXL, vae=vae, torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")
            current_model = pipe
        except Exception as e:
            print(f"Error: {e}")
            yield None, None, None, None, None, f"❌ Error: {e}", None
            return

    # 2. Generate Images
    images = {}
    directions = ["hub", "forward", "left", "right", "back"]
    
    for i, direction in enumerate(directions):
        progress((i/5), desc=f"Painting {direction.upper()}...")
        prompt = shot_list.get(direction, f"A view looking {direction}")
        img = current_model(prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        img_path = os.path.join(job_dir, f"{direction}.png")
        img.save(img_path)
        images[direction] = img_path
        
        # Yield Hub immediately when done
        if direction == "hub":
            yield images["hub"], None, None, None, None, f"✅ Saved {direction.upper()}", job_dir

    # 3. Clean up SDXL before Videos start
    print("🛑 Unloading SDXL to free VRAM for LTX...")
    unload_heavy_models() # Clears global
    if 'pipe' in locals(): del pipe # Clears local
    gc.collect()
    torch.cuda.empty_cache()

    # Final Yield: Hub is visible, Videos are empty, passing job_dir to next step
    yield images.get("hub"), None, None, None, None, "✅ Images Complete. Starting Video Engine...", job_dir


# --- Part 2: Videos Only (LTX) ---
def generate_videos(job_dir, shot_list, progress=gr.Progress()):
    """Phase 2: Generates the videos using LTX 0.9.8. Does NOT output to Hub Image."""
    global current_model
    
    if not job_dir or not os.path.exists(job_dir):
        yield None, None, None, None, "❌ Error: Job Directory missing."
        return

    # 1. Load LTX (Turbo 0.9.8)
    MODELS_ROOT = r"C:\pinokio\api\worldmAIker" # replace with your project root
    LOCAL_LTX_098_FILE = os.path.join(MODELS_ROOT, "models", "ltxv-2b-0.9.8-distilled.safetensors")
    LOCAL_T5_PATH = os.path.join(MODELS_ROOT, "models", "t5")
    
    try:
        from diffusers import LTXImageToVideoPipeline
        from transformers import T5EncoderModel, AutoTokenizer
        
        progress(0, desc="Loading LTX Engine...")
        
        # Load Components (Offline optimized)
        try:
            tokenizer = AutoTokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer", local_files_only=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")

        text_encoder = T5EncoderModel.from_pretrained(LOCAL_T5_PATH, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True)
        
        # Load Pipeline
        pipe = LTXImageToVideoPipeline.from_single_file(
            LOCAL_LTX_098_FILE,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            original_config_file=None
        )
        pipe.enable_sequential_cpu_offload() # 6GB Safety
        current_model = pipe
        
    except Exception as e:
        yield None, None, None, None, f"❌ Error loading LTX: {e}"
        return

    # 2. Generate Videos
    nav_keys = ["hub", "forward", "left", "right", "back"] 
    TOTAL_STEPS = 8 # Turbo
    video_paths = {}
    
    for direction in nav_keys:
        img_path = os.path.join(job_dir, f"{direction}.png")
        if not os.path.exists(img_path): continue
        
        # Callback for Progress Bar
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            p = (step_index + 1) / TOTAL_STEPS
            progress(p, desc=f"🎥 {direction.upper()} Step {step_index+1}/{TOTAL_STEPS}")
            return callback_kwargs

        source_img = Image.open(img_path).resize((768, 512))
        prompt = shot_list.get(direction, f"Camera moves {direction}")
        
        # --- Hub Parameter Overrides ---
        noise_scale = 0.025 # Default noise for movement
        guidance = 3.0      # Default guidance
        
        if direction == "hub":
            # Override for static anchor: high guidance, low noise
            guidance = 10.0      # Stick closely to the source image
            noise_scale = 0.005  # Minimize motion and flicker
            
        frames = pipe(
            image=source_img,
            prompt=prompt,
            height=512,
            width=768,
            num_frames=73,          
            num_inference_steps=TOTAL_STEPS, 
            guidance_scale=guidance,
            decode_timestep=0.03, 
            decode_noise_scale=noise_scale, # Uses the overridden scale
            callback_on_step_end=progress_callback
        ).frames[0]
        
        output_path = os.path.join(job_dir, f"{direction}.mp4")
        save_video_frames(frames, output_path, fps=24)
        video_paths[direction] = output_path
        
        # Yield ONLY the videos + Status (Hub Image is untouched)
        yield (
            video_paths.get("forward"), video_paths.get("left"), video_paths.get("right"), video_paths.get("back"),
            f"✅ Finished {direction.upper()}"
        )

    # 3. Cleanup
    unload_heavy_models()
    if 'pipe' in locals(): del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    yield (
        video_paths.get("forward"), video_paths.get("left"), video_paths.get("right"), video_paths.get("back"),
        "✨ World Render Complete (Turbo 0.9.8)!"
    )



def condense_video(input_path, output_dir, max_seconds=10):
    """
    Reads a video and saves a condensed version by sampling frames, aiming 
    for max_seconds length. Returns the path to the new, condensed video.
    """
    
    # Use the same FPS as the system expects (24 FPS)
    fps = 24 
    
    # 1. Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {input_path}")
        return input_path # Return original path on failure
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Target frame count for max_seconds
    target_frames = int(max_seconds * original_fps)
    
    if frame_count <= target_frames:
        cap.release()
        return input_path # Video is already short enough

    # 2. Calculate the sampling interval (e.g., if 2000 frames -> 2000/240 = 8.33, so interval is 8)
    interval = max(1, frame_count // target_frames) # Ensure interval is at least 1
    
    # 3. Prepare output video writer
    output_filename = os.path.basename(input_path).replace(".mp4", "_CONDENSED.mp4")
    output_path = os.path.join(output_dir, output_filename)
    
    # Use the original video properties for fidelity
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, original_fps, 
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # 4. Sample and write frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret: break
        
        if i % interval == 0:
            writer.write(frame)
            
    cap.release()
    writer.release()
    print(f"Condensed video saved: {output_path} (Approx {target_frames} frames)")
    
    return output_path

# --- Tab 2: Transport (The trainer) ---
def train_world_model(world_name, progress=gr.Progress()):
    """Trains the VAE and RNN using Velocity (Delta) Learning."""
    print(f"Driver: Training World Model for '{world_name}'...")
    unload_heavy_models()
    
    world_root = os.path.join(OUTPUT_DIR, world_name)
    if not os.path.exists(world_root):
        return "World not found!"

    # --- 1. DATA GATHERING ---
    training_sequences = []
    
    video_files = glob.glob(os.path.join(world_root, "**", "*.mp4"), recursive=True)
    print(f"Found {len(video_files)} videos for training.")

    for vid_path in video_files:
        filename = os.path.basename(vid_path).lower()
        
        # 1. Stronger Action Vectors (Scaled down to 2 for smoother motion)
        # --- Action Vectors ---
        if "forward" in filename:
            action_vec = [2.0, 0.0, 0.0]  # Positive Momentum
        elif "left" in filename:
            action_vec = [0.0, 2.0, 0.0]  # Rotational Change
        elif "right" in filename:
            action_vec = [0.0, 0.0, 2.0]  # Rotational Change
        elif "back" in filename:
            # Negative Momentum (Reverse)
            action_vec = [-2.0, 0.0, 0.0] 
        elif "hub" in filename:
            # Null Action (Idle/Anchor)
            action_vec = [0.0, 0.0, 0.0]  
        else:
            # Fallback for unexpected files - treat as idle
            action_vec = [0.0, 0.0, 0.0]
            
        frames = []
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
        cap.release()
        
        if len(frames) > 1:
            training_sequences.append((frames, action_vec))

    if not training_sequences:
        return "No training videos found! Generate a world first."

    # --- 2. MODEL INITIALIZATION ---
    vae_path = os.path.join(world_root, "vae.pth")
    rnn_path = os.path.join(world_root, "rnn.pth")
    
    vae = SimpleVAE().to("cuda")
    rnn = SimpleRNN().to("cuda")
    
    # Keep VAE if exists, but RETRAIN RNN FROM SCRATCH
    if os.path.exists(vae_path):
        try:
            vae.load_state_dict(torch.load(vae_path))
            print("Loaded existing VAE.")
        except:
            print("Starting VAE from scratch.")

    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    
    # --- 3. TRAINING LOOP ---
    epochs = 300 
    
    # Flatten all frames for VAE
    all_frames_rgb = []
    for seq, _ in training_sequences:
        all_frames_rgb.extend([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in seq])
    
    vae_data = torch.tensor(np.array(all_frames_rgb)).permute(0, 3, 1, 2).float() / 255.0
    vae_data = vae_data.to("cuda")
    
    batch_size = 32
    
    for epoch in progress.tqdm(range(epochs), desc="Transporting the traveler..."):
    # A. TRAIN VAE (Standard + Structural Loss)
        perm = torch.randperm(vae_data.size(0))
        epoch_loss_vae = 0
        
        for i in range(0, vae_data.size(0), batch_size):
            batch = vae_data[perm[i:i+batch_size]]
            recon, mu, logvar = vae(batch)
            
            # Reconstruction Loss
            recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction='sum')
            
            # KL Divergence Loss (Higher weight for stabilization)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total Loss: REMOVED structural_loss and INCREASED KL weight to 0.001
            loss_vae = recon_loss + (kld_loss * 0.005) 
            
            optimizer_vae.zero_grad()
            loss_vae.backward()
            optimizer_vae.step()
            epoch_loss_vae += loss_vae.item()

# ... rest of train_world_model

        # B. TRAIN RNN (VELOCITY / DELTA LEARNING)
        epoch_loss_rnn = 0
        
        for frames, action_vec in training_sequences:
            seq_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            seq_tensor = torch.tensor(np.array(seq_rgb)).permute(0, 3, 1, 2).float() / 255.0
            seq_tensor = seq_tensor.to("cuda")
            
            with torch.no_grad():
                _, mu, _ = vae(seq_tensor)
            
            if len(mu) > 1:
                current_z = mu[:-1] # Input
                next_z = mu[1:]     # Reality
                
                # TARGET is the DIFFERENCE (Velocity)
                target_delta = next_z - current_z
                
                action_tensor = torch.tensor([action_vec] * len(current_z)).float().to("cuda")
                
                # RNN predicts the DELTA
                predicted_delta, _ = rnn(current_z, action_tensor)
                
                loss_rnn = torch.nn.functional.mse_loss(predicted_delta, target_delta)
                
                optimizer_rnn.zero_grad()
                loss_rnn.backward()
                optimizer_rnn.step()
                epoch_loss_rnn += loss_rnn.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: VAE Loss={epoch_loss_vae:.2f} | RNN Loss={epoch_loss_rnn:.4f}")
            
    torch.save(vae.state_dict(), vae_path)
    torch.save(rnn.state_dict(), rnn_path)
    
    del vae, rnn, vae_data, all_frames_rgb
    cleanup_memory()
    
    return f"Training Complete! Learned VELOCITY on {len(training_sequences)} paths."

# --- Tab 3: Explore (The Dream) ---



def load_dream_stack(world_name):
    global loaded_dream_models, current_model
    
    # 1. CHECK: If the brain is already loaded, STOP HERE.
    if "vae" in loaded_dream_models and "rnn" in loaded_dream_models:
        return True 

    # 2. If we are here, the brain is missing. Time to load.
    
    # Clean up the Video Generator (Builder) if it's runningl
    if current_model is not None:
        print(f"🧹 Unloading Builder ({type(current_model).__name__}) to make room for Dreamer...")
        unload_model(current_model)
        current_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    world_dir = os.path.join(OUTPUT_DIR, world_name)
    vae_path = os.path.join(world_dir, "vae.pth") 
    
    # Explicitly check VAE file existence before loading
    if not os.path.exists(vae_path):
        print(f"ERROR: VAE brain file not found at {vae_path}")
        return False # Correctly indicates a missing file
    
    print("🧠 Loading The Eye & Traveler...")
    vae = SimpleVAE().to("cuda")
    # Explicitly map model location to 'cuda' on load
    vae.load_state_dict(torch.load(vae_path, map_location='cuda')) 
    
    rnn = SimpleRNN().to("cuda")
    # Explicitly map model location to 'cuda' on load
    rnn.load_state_dict(torch.load(os.path.join(world_dir, "rnn.pth"), map_location='cuda'))
    
    loaded_dream_models["vae"] = vae
    loaded_dream_models["rnn"] = rnn
    
    # --- HUB SEEDING (Run only once per load) ---
    print("   📍 Seeding dream with Hub Image...")
    hub_files = glob.glob(os.path.join(world_dir, "**", "hub.png"), recursive=True)
    
    if hub_files:
        img = cv2.imread(hub_files[0])
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to("cuda") / 255.0
        
        with torch.no_grad():
            _, mu, _ = vae(img_tensor)
            loaded_dream_models["state"] = mu 
    else:
        loaded_dream_models["state"] = torch.randn(1, 32).to("cuda")
        
    loaded_dream_models["hidden"] = None

    # Load Guide if needed
    if "guide" not in loaded_dream_models:
        print("👁️ Loading The Guide (Moondream)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_MOONDREAM, 
                trust_remote_code=True, 
                torch_dtype=torch.float16,
                local_files_only=True
            ).to("cuda")
        except:
            print("   🌐 Downloading Moondream...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_MOONDREAM, 
                trust_remote_code=True, 
                torch_dtype=torch.float16
            ).to("cuda")
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MOONDREAM)
        loaded_dream_models["guide"] = (model, tokenizer)

    return True # <<-- FINAL RETURN: Signaling that VAE, RNN, State, and Guide are all ready.

def dream_step(world_name, action_name):
    """The Autonomy Loop Step using Velocity predictions, now with Latent Clamping."""
    try:
        load_dream_stack(world_name)
        
        vae = loaded_dream_models["vae"]
        rnn = loaded_dream_models["rnn"]
        current_z = loaded_dream_models["state"]
        hidden = loaded_dream_models["hidden"]
        
        # Action Map (Scaled Up for Impact)
        act_map = {"forward": [2,0,0], "left": [0,2,0], "right": [0,0,2], "back": [-2,0,0]} 
        
        # Assign action_vec based on the action_name passed in
        action_vec = act_map.get(action_name, [2,0,0]) 
        action = torch.tensor([action_vec]).float().to("cuda")
        
        # --- STABILITY TUNING (Dynamic Momentum) ---
        DILATION_STEPS = 75      
        TEMPERATURE = 0.25 # was advised to change this for more movement        
        LATENT_CLAMP = 3.0       
        MOMENTUM_FACTOR = 1.05   # Default low momentum for smooth, coherent travel

        # Phase 1 ESCAPE Logic: Override momentum for the first 25 steps (managed by autonomous_dream_loop)
        if action_name == "ESCAPE":
            MOMENTUM_FACTOR = 1.8 # High momentum for quick break/traversal
            TEMPERATURE = 0.5     # HIGH noise for max exploration
        
        with torch.no_grad(): 
            for _ in range(DILATION_STEPS):
                # 1. RNN predicts VELOCITY (Delta)
                delta, hidden = rnn(current_z, action, hidden)
                
                # 2. Apply Momentum
                amplified_delta = delta * MOMENTUM_FACTOR 
                
                # 3. Apply Amplified Velocity
                next_z = current_z + amplified_delta
                
                # 4. Add Noise
                noise = torch.randn_like(next_z) * TEMPERATURE
                current_z = next_z + noise
                
                # 5. CLAMP LATENTS: Prevents collapse by enforcing bounds
                current_z = torch.clamp(current_z, -LATENT_CLAMP, LATENT_CLAMP)


            decoded_img = vae.decoder(vae.dec_fc(current_z).view(-1, 128, 8, 8))
        
        loaded_dream_models["state"] = current_z
        loaded_dream_models["hidden"] = hidden
        
        img_np = decoded_img.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).resize((512, 512), resample=Image.NEAREST)
        
        return pil_img
        
    except Exception as e:
        print(f"Dream Step Error: {e}")
        return Image.new("RGB", (512, 512), "black")



def inject_manual_question(question, current_history):
    """Saves the question to a global state for the next autonomous cycle."""
    global pending_guide_question
    
    # 1. Save the question
    pending_guide_question = question
    
    # 2. Update the chat with a confirmation message.
    # We must replace the "Guide is thinking..." placeholder from pre_update_history.
    if current_history and current_history[-1][1] == "Guide is thinking...":
        current_history[-1][1] = "Guide: **Question received.** I'll comment on this during the next cycle."
    
    # Return history and an empty string (gr.update()) for the input box
    return current_history, gr.update()

def ask_guide(image_input, question="What do you see?"):
    """Uses Moondream2 to analyze the current dream frame."""
    if image_input is None: return "I see nothing."
    
    # ... (Guide loading logic remains unchanged)
        
    if "guide" in loaded_dream_models:
        model, tokenizer = loaded_dream_models["guide"]
        
        # --- Robustly handle Gradio Input ---
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB") # Force RGB on file load
        elif isinstance(image_input, dict) and 'name' in image_input:
            image = Image.open(image_input['name']).convert("RGB") # Force RGB on file load
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB") # Ensure it's RGB if it's already PIL
        # Handle NumPy Input
        elif isinstance(image_input, np.ndarray):
            # 1. Convert numpy to PIL
            image = Image.fromarray(image_input)
            # 2. FORCE conversion to RGB for Moondream compatibility
            image = image.convert("RGB")

        else:
            print(f"Guide Input Error: Unrecognized image input type: {type(image_input)}")
            return "Guide unavailable: Image format error."
        
        enc_image = model.encode_image(image)
        answer = model.answer_question(enc_image, question, tokenizer)
        return answer
    return "Guide unavailable: Model not loaded."

# --- Logic: Wander Further / Expansion ---

def get_last_stable_frame(world_name):
    """
    Retrieves the FINAL frame from a weighted random direction in the most recent room.
    This creates branching paths (Shotgun/Tree structure) instead of a linear line.
    """
    world_dir = os.path.join(OUTPUT_DIR, world_name)
    if not os.path.exists(world_dir): return None
    
    # 1. Find the latest room folder
    subdirs = [os.path.join(world_dir, d) for d in os.listdir(world_dir)]
    rooms = [d for d in subdirs if os.path.isdir(d) and "room_" in os.path.basename(d)]
    
    if not rooms: return None
    
    rooms.sort(key=lambda x: os.path.basename(x))
    latest_room = rooms[-1]
    
    # 2. Weighted Random Direction Selection
    # We prefer forward momentum, but allow branching.
    options = ["forward", "left", "right", "back"]
    weights = [0.7, 0.1, 0.1, 0.1] # 70% Forward, 30% Turn
    
    # Check which actually exist (just in case)
    available_videos = []
    available_weights = []
    
    for opt, w in zip(options, weights):
        path = os.path.join(latest_room, f"{opt}.mp4")
        if os.path.exists(path):
            available_videos.append(path)
            available_weights.append(w)
            
    if not available_videos: return None
    
    # Normalize weights if some files are missing
    total_w = sum(available_weights)
    norm_weights = [w / total_w for w in available_weights]
    
    # Pick the winner
    video_path = np.random.choice(available_videos, p=norm_weights)
    direction_name = os.path.basename(video_path).replace(".mp4", "")
        
    print(f"🌍 Expansion Direction: {direction_name.upper()} (from {video_path})")
    
    # 3. Extract the VERY LAST frame
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert OpenCV (BGR) to PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    except Exception as e:
        print(f"Error extracting video frame: {e}")
        
    return None



def get_room_count(world_name):
    """Counts the number of successfully created room folders."""
    world_dir = os.path.join(OUTPUT_DIR, world_name)
    if not os.path.exists(world_dir):
        return 0
    
    # Count directories starting with "room_"
    rooms = [d for d in os.listdir(world_dir) if os.path.isdir(os.path.join(world_dir, d)) and "room_" in d]
    
    # Return a minimum of 1 if the world directory exists (for the initial room)
    return max(1, len(rooms))

def wander_further_prep(world_name):
    """
    Step 1 of Wander Further:
    Stops dream, unloads heavy dream models, looks at LAST REALITY frame, generates new concept.
    """
    global stop_dreaming_flag
    stop_dreaming_flag = True
    time.sleep(1) # Give loop time to die
    
    # Use stable video frame
    last_img_pil = get_last_stable_frame(world_name)
    
    if last_img_pil is None:
        return "No stable reality found (no videos). Cannot wander further.", None
    
    # 1. Ask Guide what this is (before unloading!)
    try:
        load_dream_stack(world_name) # Ensure loaded
        desc = ask_guide(last_img_pil, "You are a seasoned traveler. Describe the **SINGLE, PRIMARY LOCATION** shown in the image with vivid, sensory details including lighting. Use less than 50 words. Do not list multiple places or objects that contradict each other, unless they clearly appear in the place. Even if the place seems abstract, confidently identify the main terrain and mood.")
        
        # 2. Unload Dream Stack to free VRAM for Builder
        unload_dream_stack_only()
        
        # 3. Create context for Trip Planner
        context_prompt = f"CONTEXT: We are currently at the edge of this area: {desc}. We want to move beyond this."
        
        return f"Leaving: {desc}", context_prompt
    except Exception as e:
        print(f"Wander Prep Error: {e}")
        return "Expansion Failed", None

# --- Logic: Autonomous Dream Loop ---



def autonomous_dream_loop(world_name, existing_history=None):
    global stop_dreaming_flag
    stop_dreaming_flag = False

    world_name = world_name.strip()
    
    print(f"Starting Autonomous Dream for {world_name}...")

    # --- DYNAMIC STEP CALCULATION ---
    BASE_STEPS = 800 # Initial minimum steps
    SCALING_FACTOR = 20 # Steps added for every room created
    
    # Get the number of existing rooms for scaling
    room_count = get_room_count(world_name)
    MAX_STEPS = BASE_STEPS + (room_count * SCALING_FACTOR)
    
    print(f"Current Rooms: {room_count}. Setting Dream Limit to {MAX_STEPS} steps.")
    
    try:
        load_dream_stack(world_name)
    except FileNotFoundError:
        yield None, [["System", "Error: World data missing (Training likely failed). Check logs."]], None
        return

    timestamp = int(time.time())
    memory_path = os.path.join(OUTPUT_DIR, world_name, "memories", str(timestamp))
    os.makedirs(memory_path, exist_ok=True)
    
    step_count = 0
    chat_history = existing_history if existing_history is not None else []
    
    while not stop_dreaming_flag:
        step_count += 1
        
        # --- DYNAMIC PHASE MANAGEMENT ---
        if step_count <= 25:
            # Phase 1: High Momentum/Traversal (Triggers MOMENTUM_FACTOR = 1.8 in dream_step)
            action_name = "ESCAPE"
        else:
            # Phase 2: Smooth Exploration (Triggers MOMENTUM_FACTOR = 1.05 in dream_step)
            action_name = random.choice(["forward", "forward", "forward", "left", "right"])
            
        current_image = dream_step(world_name, action_name)
        
        filename = f"{step_count:05d}.png"
        current_image.save(os.path.join(memory_path, filename))
        
        if step_count % 7 == 0:
            # Check for pending manual question
            global pending_guide_question, guide_disabled_until
        
        current_time = time.time() # Use actual time to track the cooldown
        
        if current_time < guide_disabled_until:
            # Guide is in cooldown, skip the check completely
            injected_question = None
        else:
            # If the guide was previously disabled, try to re-enable it on the next scheduled check
            if guide_disabled_until > 0:
                print("✅ Guide Cooldown expired. Re-enabling vision check.")
                guide_disabled_until = 0 # Reset the cooldown timer
            
            # This is the normal check (step_count % 7)
            if step_count % 7 == 0:
                # Proceed with Guide query setup (manual question handling, etc.)
                # ... (rest of question setup remains the same) ...
                
                base_prompt = "You are a seasoned traveler. Describe the **SINGLE, PRIMARY LOCATION**..."
                
                if pending_guide_question:
                    # ... (manual question logic remains the same) ...
                    prompt = f"{base_prompt} CRITICAL INSTRUCTION: You must also answer this specific question based on the image: **{pending_guide_question}**"
                    injected_question = pending_guide_question 
                    pending_guide_question = ""
                else:
                    prompt = base_prompt
                    injected_question = None
            else:
                # Not the 7th step, skip Guide check
                injected_question = None

        # --- Execute Guide Query (Only if not skipped) ---
        if injected_question is not None or (step_count % 7 == 0 and guide_disabled_until == 0):
            try:
                # The Guide query must use the dynamically set prompt
                desc = ask_guide(current_image, prompt) 
            except Exception as guide_e:
                # 🛑 FAILURE: Guide Crashed - Disable for 50 steps (75 seconds)
                cooldown_time = current_time + (1.5 * 50) 
                guide_disabled_until = cooldown_time
                print(f"🛑 Guide CRASHED! Disabling vision checks until {time.ctime(cooldown_time)}.")
                desc = "Guide unavailable: Vision system failed. (Entering Cooldown)"
            
            # --- Update Chat History ---
            if injected_question:
                chat_history.append([f"Manual Query: {injected_question}", f"Guide: {desc}"])
            elif step_count % 7 == 0 and guide_disabled_until == 0:
                chat_history.append([None, f"Guide: {desc}"])
            # Note: If it crashed, we only update history with the crash message if it was a scheduled check.
            
        signal = gr.update()
        # --- Dynamic Expansion Check ---
        if step_count >= MAX_STEPS:
             chat_history.append([None, "SYSTEM: Edge of known world reached. Expanding..."])
             signal = "EXPAND"
             stop_dreaming_flag = True

        display_history = chat_history[-20:] if len(chat_history) > 20 else chat_history
            
        yield current_image, display_history, signal
        
        if signal == "EXPAND":
            break
            
        # --- Slow down to spare the browser/network ---
        if step_count < MAX_STEPS:
            time.sleep(1.5)



def launch_dream_generator(world_name, auto_checked, history):
    """Function to conditionally launch the generator and stream its output."""
    if auto_checked:
        # Launch the generator object directly
        yield from autonomous_dream_loop(world_name, history)
    else:
        # If unchecked, return an inert update for the 3 outputs
        yield gr.update(), gr.update(), gr.update()

def pre_update_history(question, current_history):
    """Immediately adds user question to history and clears input."""
    # Append the user's question with a placeholder for the answer
    current_history.append([question, "Guide is thinking..."])
    # Return the question string to be stored in the new state variable
    return current_history, "", question

def ask_guide_manual(question_from_state, current_history, dream_display_path):
    """Handles manual user queries, updates history, and calls the Guide."""
    
    # 1. The user's question is already in history as "Guide is thinking..."
    # 2. Get Guide Answer using the preserved question string
    guide_answer = ask_guide(dream_display_path, question_from_state)
    
    # 3. Find the last entry with the "Guide is thinking..." placeholder and replace it.
    # Note: We can't safely assume the last entry is the user's, so this is a simplification.
    # We will trust that the chain runs fast enough for this:
    if current_history and current_history[-1][1] == "Guide is thinking...":
        current_history[-1][1] = f"Guide: {guide_answer}"
    else:
        # Fallback if the placeholder was missed, re-add the full answer
        current_history.append([None, f"Guide: {guide_answer}"])

    # 4. Return Updated History and an inert update for the cleared input field
    return current_history, gr.update()

def reset_explore_ui(world_name):
    """
    Clears the dream display component and confirms the world name for the traveler.
    """
    return [
        gr.update(value=None), # Force the image component to clear any lingering media context
        f"Traveling in: {world_name}" # Update the status
    ]

def conditional_launch(index, world_name, auto_checked):
    """Launches dream loop only if Auto-Travel is checked and Explore tab is selected (index 2)."""
    if auto_checked and index == 2:
        # Condition met: Launch the generator
        return autonomous_dream_loop(world_name)
    else:
        # Condition not met: Return a tuple of inert updates to satisfy the 3 outputs
        # This tells Gradio to do nothing to the display, history, or signal.
        return gr.update(), gr.update(), gr.update()

def compile_sequential_dream_journal(world_name, fps=12):
    """Stitches ALL dream journal frames from ALL sessions in sequential order."""
    base_dir = os.path.join(OUTPUT_DIR, world_name, "memories")
    if not os.path.exists(base_dir): 
        return "Memory archive not found.", None
    
    # 1. Gather ALL timestamped subdirectories (dream sessions)
    # The sort order (alphabetical/numeric) of the timestamped folders naturally gives us the chronological order.
    runs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    if not runs: 
        return "No dream sessions found.", None
        
    print(f"Compiling memories from {len(runs)} sessions...")
    
    images = []
    total_frames = 0
    
    # 2. Iterate through ALL run folders in chronological order
    for run in runs:
        run_path = os.path.join(base_dir, run)
        
        # Get all frames within that session, sorted by step count (e.g., 00001.png, 00002.png)
        files = sorted([f for f in os.listdir(run_path) if f.endswith('.png')])
        
        for f in files:
            # Append frame to the list
            images.append(imageio.imread(os.path.join(run_path, f)))
            total_frames += 1

    if not images: 
        return "No frames found.", None
    
    # 3. Assemble the Final Video
    # Use the timestamp of the LATEST run for the output file name
    output_video = os.path.join(OUTPUT_DIR, world_name, f"FULL_journal_{runs[-1]}.mp4")
    imageio.mimwrite(output_video, images, fps=fps, codec="libx264")
    
    return output_video, f"Full Dream Journal Compiled ({total_frames} frames from {len(runs)} sessions)"



def format_video_for_html(video_path, width="100%"):
    """Formats a video path for safe Gradio HTML playback."""
    if not video_path: return ""
    
    # --- Make path relative to Gradio's server root ---
    # Find the index of the first 'outputs' folder
    outputs_index = video_path.find('outputs')
    
    if outputs_index == -1:
        # Fallback if the path structure is unexpected (shouldn't happen)
        gradio_src = f"/file={video_path}"
    else:
        # Use only the relative path starting from 'outputs'
        # e.g., /file=outputs/Autonomous_.../FULL_journal.mp4
        relative_path = video_path[outputs_index:]
        gradio_src = f"/file={relative_path}" 
        
    return f"""
    <video width="{width}" controls autoplay muted loop>
        <source src="{gradio_src}" type="video/mp4">
        Your browser does not support the video tag.
    </video>"""

# ... (Lines 945-988 for downscale_and_display)



def make_native_video_html(path):
    """Generates the custom HTML for the 256x256 pixelated player using the secure Gradio file path."""
    if not path:
        return ""
    # We use the path provided by the gr.File component, which is the secure, public URL.
    return f"""
    <div style="border: 1px solid #ddd; width: 70px; height: 70px; display: flex; align-items: center; justify-content: center; background-color: #333;">
        <video id="native_video" width="64" height="64" controls autoplay muted loop style="image-rendering: pixelated;">
            <source src="{path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <p style="text-align: center; margin-top: 5px;">Playback (64x64 Native)</p>
    """

def start_compilation_feedback():
    """Immediately updates the UI to show compilation has started."""
    return [
        gr.update(value="⏳ Compiling...", interactive=False, variant="secondary"), # Disable and change button
        "🎬 Starting Dream Journal Compilation..." # Update status text
    ]

def downscale_and_display(video_path, status_message):
    """
    Downscales the source video to 64x64 and returns the HTML for native display.
    Requires imageio (which uses ffmpeg) to be installed.
    """
    if not video_path or "Error" in status_message:
        # Ensure the function returns 2 values even in error case
        return None, status_message 
        
    # --- FIND THE CORRECT OUTPUT DIRECTORY ---
    # The video path is in the format: C:\...\outputs\WORLD_NAME\FULL_journal_...mp4
    # We want the directory containing that file (i.e., the WORLD_NAME folder)
    
    # 1. Get the directory containing the video file
    output_dir = os.path.dirname(video_path) 
    
    # 2. Define a new filename for the downscaled video
    base_name = os.path.basename(video_path).replace(".mp4", "_64x64.mp4")
    output_path = os.path.join(output_dir, base_name) # Saves the new file in the World's root
    
    print(f"Downscaling {video_path} to 64x64, saving to {output_path}...")
    
    # --- SERVER-SIDE RESIZING USING FFMPEG/imageio ---
    try:
        # Use imageio or another library to handle the actual resizing/conversion
        
        reader = imageio.get_reader(video_path)
        writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'])
        
        for i, frame in enumerate(reader):
            # Convert frame to PIL, resize, then convert back to numpy
            pil_frame = Image.fromarray(frame).resize((64, 64), resample=Image.NEAREST)
            writer.append_data(np.array(pil_frame))
            
        writer.close()
        reader.close()
        
    except Exception as e:
        error_message = f"Downscale failed: {e}"
        # Ensure the function returns 2 values on failure
        return None, error_message 

    # --- Generate HTML for 64x64 Native Display ---
    # Make path relative for HTML display
    outputs_index = output_path.find('outputs')
    relative_path = output_path[outputs_index:]
    gradio_src = f"/file={relative_path}"
        
    # Upscale container and video tags to 256x256 (4x scale) for visible controls
    video_html = f"""
    <div style="border: 1px solid #ddd; width: 258px; height: 258px; display: flex; align-items: center; justify-content: center; background-color: #333;">
        <video id="native_video" width="256" height="256" controls autoplay muted loop style="image-rendering: pixelated;">
            <source src="{gradio_src}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <p style="text-align: center; margin-top: 5px;">Playback (64x64 Native)</p>
    """
    
    return output_path, "Starting 64x64 Native Playback..."

# --- UI Helpers (NEW) ---

def update_world_name_dynamic(vibe):
    """Updates filename based on dropdown change (New Feature)."""
    clean_vibe = vibe.replace(" ", "_")
    return f"{clean_vibe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

def update_header(world_name):
    return f"### 🗺️ Coordinates: {world_name}"

# --- UI Layout ---

def toggle_manual_mode(is_manual):
    """
    Toggles the UI between Autonomous and Manual Training modes.
    """
    return [
        # 1. Disable/Enable Autonomous Buttons
        gr.update(interactive=not is_manual), # plan_btn
        gr.update(interactive=not is_manual), # shot_list_btn
        gr.update(interactive=not is_manual), # render_btn
        gr.update(interactive=not is_manual), # full_auto_btn
        
        # 2. Show/Hide Manual Commit Button
        gr.update(visible=is_manual), # commit_btn
        
        # 3. Show/Hide Manual Inputs (Hub Video)
        gr.update(visible=is_manual), # manual_hub_video
        
        # 4. Toggle Interactivity of Media Components
        gr.update(interactive=is_manual), # hub_image_out
        gr.update(interactive=is_manual), # video_fwd
        gr.update(interactive=is_manual), # video_left
        gr.update(interactive=is_manual), # video_right
        gr.update(interactive=is_manual), # video_back
        
        # 5. Disable Wander Button (Tab 3)
        gr.update(interactive=not is_manual), # wander_btn
        
        # 6. Status Message
        "Manual Mode Active. Autonomous generation disabled." if is_manual else "Autonomous Mode Active."
    ]

def commit_manual_room(world_name, hub_img, hub_vid, vid_fwd, vid_left, vid_right, vid_back):
    """
    Saves manually uploaded files to a new room directory.
    """
    if not world_name:
        return "Error: World Name is required."
    
    # Explicitly check for raw NumPy array input
    file_inputs = [hub_img, hub_vid, vid_fwd, vid_left, vid_right, vid_back]

    for input_file in file_inputs:
        if isinstance(input_file, np.ndarray):
            # This input came from a gr.Image component returning raw data.
            return "Error: Hub Image input format is invalid. Please ensure all media are uploaded as file paths (not raw image data or drawing)."
    
    # Original validation check (now safe from NumPy array crash)
    if not all(file_inputs):
        return "Error: All 6 media files (Hub Image + 5 Videos) are required."

    # Create Room Directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    room_dir = os.path.join(OUTPUT_DIR, world_name, f"room_MANUAL_{timestamp}")
    os.makedirs(room_dir, exist_ok=True)
    
    try:
        # Copy Files
        # Hub Image
        shutil.copy(hub_img, os.path.join(room_dir, "hub.png"))

        video_map = {
            "hub": hub_vid, 
            "forward": vid_fwd, 
            "left": vid_left, 
            "right": vid_right, 
            "back": vid_back
        }

        condensed_video_paths = {}
        for direction, path in video_map.items():
            # Pass the path and the destination (room_dir) to the condensation function
            # The condensed_path is the new source of truth.
            condensed_path = condense_video(path, room_dir, max_seconds=10) 
            condensed_video_paths[direction] = condensed_path
        
        # Videos (Rename to match training expectations)
        shutil.copy(condensed_video_paths["hub"], os.path.join(room_dir, "hub.mp4"))
        shutil.copy(condensed_video_paths["forward"], os.path.join(room_dir, "forward.mp4"))
        shutil.copy(condensed_video_paths["left"], os.path.join(room_dir, "left.mp4"))
        shutil.copy(condensed_video_paths["right"], os.path.join(room_dir, "right.mp4"))
        shutil.copy(condensed_video_paths["back"], os.path.join(room_dir, "back.mp4"))
        
        return f"✅ Manual Room Committed & Condensed (Max 10s per video): {room_dir}", world_name
        
    except Exception as e:
        return f"Error committing room: {str(e)}"

glass_theme = gr.themes.Glass()

with gr.Blocks(theme=glass_theme, title="worldmAIker") as app:
    gr.Markdown("# 🌍 worldmAIker: Autonomous World Traveler v4 [autonomous or manual training]")
    existing = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))] if os.path.exists(OUTPUT_DIR) else []
    
    # 1. Determine the last used world name (for Explore/Transport default)
    existing.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    last_existing_world = existing[0] if existing else "my_new_world"
    
    # 2. Determine the default name for a new build (always unique)
    default_vibe = "Autonomous"
    new_build_name = f"{default_vibe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Set initial state for input boxes (The name to be used for the current run)
    initial_world_name_build = new_build_name
    # Set initial state for the Explore/Transport states (The name they should load if they start there)
    initial_world_state = last_existing_world

    with gr.Tabs() as tabs:
        # --- TAB 1: BUILD ---
        with gr.Tab("Plan Travel"):
            with gr.Row():
                with gr.Column():
                    default_vibe = "Autonomous"
                    vibe_input = gr.Dropdown(["Autonomous", "Cyberpunk City", "Misty Forest", "Pretty Beach", "Underwater Ruins", "Desert Oasis", "Mars Colony", "Mountain Town"], label="Vibe", value=default_vibe)
                    custom_prompt = gr.Textbox(label="Or Custom Prompt / Context", placeholder="Leave empty for full autonomy...")
                    
                    # --- Manual Mode Toggle ---
                    manual_mode_chk = gr.Checkbox(label="Manual Training Mode", value=False)
                    
                    plan_btn = gr.Button("Travel Agent", variant="secondary")
                    location_desc = gr.Textbox(label="Location Description", lines=3, interactive=False)
            
            with gr.Row():
                with gr.Column():
                    shot_list_btn = gr.Button("Plan Navigation")
                    shot_list_json = gr.JSON(label="Shot List")
                
                    world_name_input = gr.Textbox(label="World Name", value=initial_world_name_build)
                    render_btn = gr.Button("Visualize Journey", variant="stop")
                    # --- Manual Commit Button ---
                    commit_btn = gr.Button("Commit Manual Room", variant="primary", visible=False)
            
# ... inside "Plan Travel" tab ...
            with gr.Row():
                hub_image_out = gr.Image(label="Hub Image", interactive=False, type="filepath")
                
                # --- Manual Hub Video Input ---
                manual_hub_video = gr.Video(label="Hub Video (Manual Only)", visible=False)
                
                video_fwd = gr.Video(label="Forward", interactive=False)
                video_left = gr.Video(label="Left", interactive=False)
                video_right = gr.Video(label="Right", interactive=False)
                video_back = gr.Video(label="Back", interactive=False)

            with gr.Row():
                full_auto_btn = gr.Button("🚀 AUTONOMOUS JOURNEY...", variant="primary")

            with gr.Row():
                auto_status = gr.Textbox(label="Current Status", lines=1, interactive=False)
            
            # --- State for Chaining ---
            job_dir_state = gr.State()

            plan_btn.click(generate_trip_concept, inputs=[vibe_input, custom_prompt], outputs=location_desc)
            shot_list_btn.click(generate_shot_list, inputs=location_desc, outputs=shot_list_json)
            
            # --- Rendering Chain ---
            # 1. Generate Images -> Updates Hub -> Returns Job Dir
            render_btn.click(
                generate_images, 
                inputs=[shot_list_json, world_name_input], 
                outputs=[hub_image_out, video_fwd, video_left, video_right, video_back, auto_status, job_dir_state]
            ).then(
                # 2. Generate Videos -> Updates Videos -> Reads Job Dir
                generate_videos,
                inputs=[job_dir_state, shot_list_json],
                outputs=[video_fwd, video_left, video_right, video_back, auto_status]
            )
            
            def chain_start(vibe, prompt, name): return "Logistical planning..."

        # --- TAB 2: TRAIN ---
        with gr.Tab("Transport"):
            # Use the last_existing_world as the default value here
            train_world_name = gr.Textbox(label="World Selection", value=initial_world_state)
            train_btn = gr.Button("Transport", variant="stop")
            train_log = gr.Textbox(label="Transport Log")
            train_btn.click(train_world_model, inputs=train_world_name, outputs=train_log)

        # --- TAB 3: EXPLORE ---
        with gr.Tab("Explore"):
            # Use the last_existing_world as the default value for the state
            current_world_state = gr.State(initial_world_state) 
            expansion_job_state = gr.State()
            last_compiled_video_path = gr.State()
            # REMOVED downscaled_video_path = gr.State() 
            is_small_visible_state = gr.State(False)
            
            downscaled_video_file = gr.File(type="filepath", visible=False)
            native_html_content = gr.State(value="")
            manual_query_state = gr.State(value="")

            with gr.Row():
                world_header = gr.Markdown(f"### 🗺️ Coordinates: {initial_world_state}")

            with gr.Row():
                with gr.Column(scale=2):
                    dream_display = gr.Image(label="Traveling...")
                    with gr.Row():
                        btn_left = gr.Button("⬅️")
                        btn_fwd = gr.Button("⬆️")
                        btn_back = gr.Button("⬇️")
                        btn_right = gr.Button("➡️")
                    with gr.Row():
                        auto_dream_chk = gr.Checkbox(label="Auto-Travel", value=False)
                        wander_btn = gr.Button("🥾 WANDER FURTHER (Expand World)", variant="secondary")
                    edge_signal = gr.Textbox(visible=False, label="Edge Signal")
                    
                with gr.Column(scale=1):
                    chat_history = gr.Chatbot(label="Travel Guide")
                    
                    # --- UI Components for Manual Query ---
                    guide_query_input = gr.Textbox(label="Ask The Guide", placeholder="e.g., Do you see a rock here? What color is the sky?")
                    guide_query_btn = gr.Button("Ask", variant="secondary")
                    # --- End Manual Query Components ---
                    
                    gr.Markdown("### Memories")
                    compile_btn = gr.Button("🎬 Compile World Dream Journal", variant="primary")
                    toggle_size_btn = gr.Button("🔄 Toggle Playback Size", variant="stop") 
                        
                    # --- Large Player ---
                    video_out_large = gr.Video(label="Playback (Full Scale)", visible=False)                    
                    # --- Native 64x64 Player ---
                    # CHANGED gr.Video to gr.HTML for native sizing
                    video_out_native_html = gr.HTML(value="", visible=False)
            
            current_world_state.change(update_header, inputs=current_world_state, outputs=world_header)

            # This is the most widely tested structure for passing a state variable and a static string:
            btn_fwd.click(lambda w: dream_step(w, "forward"), inputs=current_world_state, outputs=dream_display)
            btn_left.click(lambda w: dream_step(w, "left"), inputs=current_world_state, outputs=dream_display)
            btn_right.click(lambda w: dream_step(w, "right"), inputs=current_world_state, outputs=dream_display)
            btn_back.click(lambda w: dream_step(w, "back"), inputs=current_world_state, outputs=dream_display)

            guide_query_btn.click(
                # STEP 1: Immediate update to show the user's question and clear the input
                pre_update_history,
                # Inputs: Question text, current history
                inputs=[guide_query_input, chat_history], 
                # Outputs: Updated history, cleared input, and question string is NOT saved to state anymore
                outputs=[chat_history, guide_query_input, manual_query_state],
                queue=False # Force this UI update immediately
            ).then(
                # STEP 2: FAST INJECTION (Replaces the slow LLM call)
                inject_manual_question,
                # Inputs: The question from the state (which still holds the string), current history
                inputs=[manual_query_state, chat_history], 
                # Outputs: Final updated history, and an inert update for the input box
                outputs=[chat_history, guide_query_input], 
                queue=False # Force this UI update immediately
            )

            # Compile Button Logic
            compile_btn.click(
                # --- Step 0: Immediate Feedback ---
                # Call the new helper to disable the button and set status
                start_compilation_feedback, 
                inputs=[], # No inputs needed for this function
                outputs=[compile_btn, auto_status] # Outputs are the button itself and the status
            ).then(
                # 1. Compile the video (Outputs: [path, status_message])
                compile_sequential_dream_journal, 
                inputs=current_world_state, 
                # Store the path in the state variable for later use
                outputs=[last_compiled_video_path, auto_status] 
            ).then(
                # 2. IMMEDIATE STATUS UPDATE: Prepare to downscale (Instant)
                lambda status: "Starting 64x64 Downscale...",
                inputs=[auto_status],
                outputs=[auto_status]
            ).then(
                # 3. Downscale the video. Now returns the PATH string (not HTML)
                downscale_and_display, 
                inputs=[last_compiled_video_path, auto_status], # Path and Status for error checking
                outputs=[downscaled_video_file, auto_status]
            ).then(
                # 4. Final Display: Show full path, hide HTML, and reset toggle state
                # Re-enable the button and set final success text
                lambda path_full, status: [
                    gr.update(value=path_full, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="🎬 Compile World Dream Journal", interactive=True, variant="primary"), # RE-ENABLE BUTTON
                    "✨ Dream Journal Ready for Playback.", # FINAL STATUS
                    False 
                ],
                # Inputs: Only need the full path state and status here
                inputs=[last_compiled_video_path, auto_status], 
                outputs=[video_out_large, video_out_native_html, compile_btn, auto_status, is_small_visible_state], 
                queue=False
            )

            downscaled_video_file.change(
                fn=make_native_video_html,
                inputs=[downscaled_video_file], # Receives the secure URL
                outputs=[native_html_content],   # Stores the generated HTML string
                queue=False
            )
            
            # Wiring for downscale_btn
            toggle_size_btn.click(
                # The lambda receives the current toggle state and the pre-rendered HTML content.
                lambda is_small_active, html_content: [
                    # Action 1: Update LARGE Player. 
                    gr.update(visible=is_small_active), 
                    # Action 2: Update NATIVE HTML Player. Set content and toggle visibility.
                    gr.update(value=html_content, visible=not is_small_active),
                    # Action 3: Return the new toggled state
                    not is_small_active # Toggle the state
                ],
                # Inputs: Toggle state and the HTML content state variable
                inputs=[is_small_visible_state, native_html_content], 
                outputs=[video_out_large, video_out_native_html, is_small_visible_state],
                queue=False
            )
    # --- CHAINING LOGIC ---
    
    # --- WIRING FIX 2: Full Auto Chain ---
    # Added auto_status to the render_world outputs list
    # --- CHAINING LOGIC (Update this part too) ---
    # --- Full Auto Chain ---
    full_auto_btn.click(
        chain_start, inputs=[vibe_input, custom_prompt, world_name_input], outputs=auto_status
    ).then(
        generate_trip_concept, inputs=[vibe_input, custom_prompt], outputs=location_desc
    ).then(
        generate_shot_list, inputs=location_desc, outputs=shot_list_json
    ).then(
        # Phase 1: Images (Returns 7 outputs...)
        generate_images, 
        inputs=[shot_list_json, world_name_input], 
        outputs=[hub_image_out, video_fwd, video_left, video_right, video_back, auto_status, job_dir_state]
    ).then(
        # LAMBDA: Takes all 7 outputs, returns only the job_dir string.
        lambda hub, fwd, left, right, back, status, job_dir: job_dir, 
        inputs=[hub_image_out, video_fwd, video_left, video_right, video_back, auto_status, job_dir_state],
        outputs=[job_dir_state],
        queue=False 
    ).then(
        # Phase 2: Videos (Reads job_dir_state and shot_list_json.value)
        generate_videos,
        inputs=[job_dir_state, shot_list_json], 
        outputs=[video_fwd, video_left, video_right, video_back, auto_status]
    ).then(
        lambda: "Training...", None, auto_status
    ).then(
        # 3. Train
        train_world_model, inputs=world_name_input, outputs=auto_status
    ).then(
        # 4. Final Step: Auto-Launch Logic
        lambda world_name: [
            reset_explore_ui(world_name)[0], # Clear dream_display
            f"✅ World built and trained: {world_name}. Traveling...", # New Status
            world_name, # New World Name for the State variable
            True # Checkbox value (True = Auto-Launch)
        ], 
        inputs=[world_name_input],
        outputs=[
            dream_display, 
            auto_status, 
            current_world_state,
            auto_dream_chk
        ]
    )


    # --- Expansion Logic ---
    def expansion_chain_logic(trigger_component):
        return trigger_component.then(
            lambda: "Analyzing Memories & Expanding...", None, auto_status
        ).then(
            # 1. Prep
            wander_further_prep, inputs=current_world_state, outputs=[auto_status, custom_prompt]
        ).then(
            # 2. Plan
            generate_trip_concept, inputs=[vibe_input, custom_prompt], outputs=location_desc
        ).then(
            # 3. Shot List
            generate_shot_list, inputs=location_desc, outputs=shot_list_json
        ).then(
            # 4. Phase 1: Images (Returns 7 outputs...)
            generate_images, 
            inputs=[shot_list_json, current_world_state], 
            outputs=[hub_image_out, video_fwd, video_left, video_right, video_back, auto_status, expansion_job_state]
        ).then(
            # LAMBDA: Takes all 7 outputs, returns only the job_dir string.
            lambda hub, fwd, left, right, back, status, job_dir: job_dir, 
            inputs=[hub_image_out, video_fwd, video_left, video_right, video_back, auto_status, expansion_job_state],
            outputs=[expansion_job_state],
            queue=False
        ).then(
            # 5. Phase 2: Videos (Reads expansion_job_state and shot_list_json.value)
            generate_videos,
            inputs=[expansion_job_state, shot_list_json], 
            outputs=[video_fwd, video_left, video_right, video_back, auto_status]
        ).then(
            lambda: "Retraining World Model...", None, auto_status
        ).then(
            # 6. Train
            train_world_model, inputs=current_world_state, outputs=auto_status
        ).then(
        lambda: "Resuming Journey...", None, auto_status
        ).then(
            # NEW STEP: Clear the display before the dream loop resumes

            lambda world_name, history: [reset_explore_ui(world_name)[0], history], 
            inputs=[current_world_state, chat_history],
            outputs=[dream_display, chat_history],
            queue=False
        ).then(
            # 7. Resume Dreaming

            autonomous_dream_loop, 
            inputs=[current_world_state, chat_history],
            outputs=[dream_display, chat_history, edge_signal]
        )

    expansion_chain_logic(wander_btn.click(lambda: None, None, None))
    expansion_chain_logic(
        edge_signal.change(
            # New FN: If signal is "EXPAND", return the world name (current_world_state).
            # Change gr.NoOp() to gr.skip()
            lambda sig, world_name: world_name if sig == "EXPAND" else gr.skip(),
            inputs=[edge_signal, current_world_state],
            # Output MUST match the input to the next step (wander_further_prep).
            outputs=[current_world_state] # Pass the world name to the next chain
        )
    )

    train_world_name.change(lambda x: x, inputs=train_world_name, outputs=current_world_state)
    vibe_input.change(update_world_name_dynamic, inputs=vibe_input, outputs=[world_name_input, train_world_name, current_world_state])

    auto_dream_chk.change(
    # This block handles the synchronous reset AND the conditional launch
    fn=launch_dream_generator,
    inputs=[current_world_state, auto_dream_chk, chat_history],
    outputs=[dream_display, chat_history, edge_signal],
    queue=True
)

    # --- Manual Mode Wiring ---
    manual_mode_chk.change(
        toggle_manual_mode,
        inputs=[manual_mode_chk],
        outputs=[
            plan_btn, shot_list_btn, render_btn, full_auto_btn, # Auto Buttons
            commit_btn, # Manual Commit Button
            manual_hub_video, # Manual Hub Video
            hub_image_out, video_fwd, video_left, video_right, video_back, # Media Components
            wander_btn, # Tab 3 Button (Global)
            auto_status # Status Message
        ]
    )

    tabs.select(
    fn=lambda name: name, # Simple lambda to pass the input value through
    inputs=[world_name_input], # Get the latest name from the input box
    outputs=[current_world_state], # Write it to the Explore State component
    queue=False
    )
    
    commit_btn.click(
    commit_manual_room,
    inputs=[world_name_input, hub_image_out, manual_hub_video, video_fwd, video_left, video_right, video_back],
    # Outputs status, Transport name, and Explore state
    outputs=[auto_status, train_world_name] 
    # Note: train_world_name.change already wires to current_world_state
    )

if __name__ == "__main__":
    app.launch()