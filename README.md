# F5-TTS Voice Cloning Demo

A small Gradio interface for zero-shot voice cloning with
[F5-TTS](https://github.com/SWivid/F5-TTS). Record a short reference clip and
synthesize new text in the reference voice. The recording is transcribed
automatically and deleted after generation.

Only use audio you own or have permission to use. Do not use generated speech
to impersonate, deceive, or harm anyone.

## Run with the existing environment

```bash
cd /home-nfs/jjery2243542/voice_cloning
conda activate f5-tts
python app_f5.py
```

Open <http://127.0.0.1:7861>. The model is downloaded on first use, so the first
generation takes longer. A CUDA GPU is strongly recommended.

If you are connected to this machine over SSH, forward the port from your local
computer:

```bash
ssh -L 7861:localhost:7861 USER@SERVER
```

Then open <http://127.0.0.1:7861> locally.

### Public Gradio link (`share=True`)

To enable Gradio sharing and create a temporary public URL, run:

```bash
SHARE=1 python app_f5.py
```

The app reads `SHARE=1` and passes `share=True` to Gradio. The terminal will
print a public `https://...gradio.live` URL. Anyone with that URL can access the
app while the process is running, so share it carefully.

Use a different port with `PORT=9000 python app_f5.py`.

## Install in a new environment

Install `ffmpeg` first, then run:

```bash
conda create -n f5-tts python=3.10 -y
conda activate f5-tts
pip install -r requirements.txt
python app_f5.py
```

PyTorch installation differs by CPU/GPU platform. If the command above does not
install the correct build, install PyTorch and torchaudio using the selector at
<https://pytorch.org/get-started/locally/>, then rerun the requirements command.

For development against a local F5-TTS checkout instead of the installed
package:

```bash
F5_TTS_PATH=/path/to/F5-TTS python app_f5.py
```

## Create the GitHub repository

The `.gitignore` excludes generated audio, spectrograms, caches, notebooks, and
local reference recordings.

First create a new **empty** repository named `f5-voice-cloning-demo` at
<https://github.com/new>. Do not add a README, `.gitignore`, or license there,
because those files already exist locally.

This managed workspace currently has an empty, read-only `.git` placeholder.
In your regular terminal, remove that empty directory first (`rmdir` will refuse
to remove it if it ever contains files), then initialize and push the repository:

```bash
rmdir .git
git init
git add app_f5.py README.md requirements.txt .gitignore
git commit -m "Initial F5-TTS voice cloning demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/f5-voice-cloning-demo.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Choose **Private** when
creating the repository unless you are ready to publish the code openly.
