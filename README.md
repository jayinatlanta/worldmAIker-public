# 🌍 worldmAIker: Infinite Autonomous World Builder

**worldmAIker** is the world's first free, fully autonomous, recursive, local AI world-building and exploration engine. It doesn't just generate scenes; it *dreams* them, travels through them, and physically constructs them in real-time using a symphony of state-of-the-art local models.

From a single prompt, it hallucinates a 360° reality, renders it, animates it, learns its physics, and then uses that knowledge to dream up the next horizon—infinitely. It is a self-contained, creative singularity running on a relatively small GPU (~3GB free VRAM).

You can let it run automatically to discover new lands (Autonomous Mode), or manually train it with your own footage to create a digital twin of a real-world location (Director Mode).

This is a small but powerful application for understanding LLMs, world models, and multi-agent systems, the most important components of AI today.

---

## ✨ Features

*   **Infinite World Generation:** autonomously plans trips, generates shot lists, and renders 360° views of new locations.
*   **Autonomous Travel:** A "Traveler" agent (RNN) navigates the latent space of the world, making decisions on where to go next.
*   **Dream Journaling:** Records every step of the journey and compiles them into a cohesive video journal.
*   **The Guide:** A Vision-Language Model (VLM) that watches the journey with you and can answer questions about the scenery.
*   **Manual Training Mode:** Upload your own videos/images to train the world model on specific locations.
*   **Wander Further:** When you reach the edge of the known world, the system intelligently expands the map based on the visual context.

---

## 🛠️ Prerequisites & Models

**⚠️ IMPORTANT:** This system runs entirely locally. You are responsible for downloading the necessary models, and are then subject to the terms of their licenses.

### 1. Ollama (Required)
You must have [Ollama](https://ollama.com/) installed and running.
*   **Model:** `gemma3:4b`
*   **Action:** Run the following command in your terminal:
    ```bash
    ollama pull gemma3:4b
    ```

### 2. Local Models (Hugging Face)
The system uses several models from Hugging Face. The first run will download significant data (several GBs).
*   **Image Generation:** `stabilityai/sdxl-turbo` (SDXL Turbo)
*   **Vision/Guide:** `vikhyatk/moondream2` (Moondream 2)

### 3. LTX Video Model (Critical Dependency)
The video generation engine relies on **LTX Video**.
*   **Path:** The system currently looks for the model at:
    `C:\pinokio\api\filmmAIker.git\models\ltxv-2b-0.9.8-distilled.safetensors`
*   **Action:** Ensure you have the `filmmAIker` project installed in Pinokio, or manually place the `ltxv-2b-0.9.8-distilled.safetensors` model in that specific path.

### 4. FFmpeg
Required for video processing. The setup script attempts to install it via `winget`, but ensure it is in your system PATH.

---

## 🚀 User Guide

### Mode 1: Autonomous Creation (The "God" Mode)
Let the AI build the world for you.

0.  **Complete Autonomy:** All you NEED to do is simply click **🚀 AUTONOMOUS JOURNEY**, and the system will create its own world, and explore, and then create new locations in that world, forever. You can check in with its exploration on the **Explore** tab.

But...if you want some input....

1.  **Plan Travel:**
    *   Go to the **Plan Travel** tab.
    *   Select a **Vibe** (e.g., "Cyberpunk City", "Misty Forest") or enter a **Custom Prompt**.
    *   Click **Travel Agent** to generate a location description.
2.  **Plan Navigation:**
    *   Click **Plan Navigation** to generate a "Shot List" (JSON) defining the 360° views (Hub, Forward, Left, Right, Back).
3.  **Visualize:**
    *   Click **Visualize Journey**.
    *   The system will generate the static images (SDXL) and then animate them into videos (LTX).
4.  **Train:**
    *   Go to the **Transport** tab.
    *   Click **Transport** to train the World Model (VAE + RNN) on the newly generated videos.
5.  **Explore:**
    *   Go to the **Explore** tab.
    *   Click **Auto-Travel** to let the agent wander, or use the arrow buttons to move manually.

### Mode 2: Manual Training (The "Director" Mode)
Train the AI on your own footage.

1.  **Activate Manual Mode:**
    *   In the **Plan Travel** tab, check the **Manual Training Mode** box.
2.  **Upload Media:**
    *   **Hub Image:** Upload a high-quality static image of your location.
    *   **Videos:** Upload 5 videos corresponding to the directions: **Hub** (360/Loop), **Forward**, **Left**, **Right**, **Back**.
3.  **Commit:**
    *   Enter a **World Name**.
    *   Click **Commit Manual Room**. The system will process and condense your videos (max 10s).
4.  **Train:**
    *   Go to the **Transport** tab and click **Transport** to train the model on your data.

### Mode 3: The Guide & Dreaming
Interact with the world while exploring.

*   **Ask the Guide:** In the **Explore** tab, type a question (e.g., "What architectural style is this?") and click **Ask**. The VLM will analyze the current view and respond.
*   **Wander Further:** If you reach the edge of the generated content, click **WANDER FURTHER**. The system will look at the last frame, dream up a new adjacent location, and expand the world.
*   **Compile Journal:** Click **Compile World Dream Journal** to stitch your entire journey into a single movie file.

---

## ⚖️ License & Content Disclaimer

**User Responsibility:**
You are solely responsible for the content you generate using this tool. Ensure you have the rights to use any manual training data (images/videos) you upload.

**Third-Party Licenses:**
This project utilizes several open-source models, each with its own license. You must comply with them:
*   **SDXL Turbo:** [SA-HO Model License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE) (Non-Commercial Research Community License).
*   **LTX Video:** [OpenRAIL](https://huggingface.co/Lightricks/LTX-Video) (or specific license provided by Lightricks).
*   **Moondream2:** Apache 2.0.
*   **Gemma 3:** [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

By using this software, you agree to adhere to these licenses and use the technology ethically and legally.

---

## 📦 Repository Structure (Required Files)

If you are forking or distributing this project, ensure the following files are included for the system to function correctly within Pinokio:

*   **`LICENSE`**: Required license file.
*   **`.gitignore`**: Crucial for keeping the repo clean and excluding large model files.
*   **`app.py`**: The core application logic.
*   **`requirements.txt`**: Python dependencies.
*   **`setup.bat`**: Installation and environment setup script.
*   **`README.md`**: This documentation file.
*   **`pinokio.js`**: Pinokio menu configuration and metadata.
*   **`install.js`**: Pinokio install script.
*   **`start.js`**: Pinokio launch script.
*   **`icon.png`**: Application icon.

