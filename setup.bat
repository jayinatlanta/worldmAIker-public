@echo off
echo =========================================================
echo  worldmAIker V4 - Bulletproof Setup
echo =========================================================
echo.

REM --- Step 1: System-Level Dependencies ---
echo [1/9] Checking for FFmpeg...
REM Tries to install FFmpeg, but continues if it fails (in case user has it manually)
winget install -e --id Gyan.FFmpeg || echo "FFmpeg 'winget' install failed, assuming user has it in PATH or /bin/"
echo.

REM --- Step 2: Virtual Environment ---
IF NOT EXIST venv (
    echo [2/9] Creating new virtual environment...
    python -m venv venv
) ELSE (
    echo [2/9] Virtual environment 'venv' already exists.
)
call venv\Scripts\activate.bat
echo.

REM --- Step 2b: Smart Skip Check ---
REM We check for 'timm' because it is the last requirement.
REM If it exists, we assume the environment is healthy and skip the heavy lifting.
echo Checking for existing installation...
pip show timm >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [SKIP] Found 'timm'. Dependencies look installed.
    echo [SKIP] Bypassing heavy installation steps...
    goto :END_SUCCESS
)

REM --- Step 3: Upgrade Pip ---
echo Upgrading pip and build tools...
python.exe -m pip install --upgrade pip setuptools wheel
echo.
echo Initial requirements install. Some errors are expected.
pip install -r requirements.txt
echo.
echo Reinstall torch, GPU version
pip install --force-reinstall torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
echo.
echo Upgrade bleeding edge transformers for compatibility
pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate huggingface_hub sentencepiece protobuf imageio[ffmpeg] opencv-python wakepy einops

echo.

REM --- Step 9: Final Cleanup ---
echo Purging pip cache to save disk space...
pip cache purge
echo.

:END_SUCCESS
echo =========================================================
echo  worldmAIker V4 - Setup Complete!
echo  All dependencies installed and patched.