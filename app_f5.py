from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

OUTPUTS_DIR = ROOT / "outputs_f5"
SPECTROGRAMS_DIR = ROOT / "spectrograms_f5"

# `pip install f5-tts` is the normal setup. During F5-TTS development, this
# optional environment variable can point at a local checkout instead.
F5_TTS_PATH = os.environ.get("F5_TTS_PATH")
if F5_TTS_PATH:
    f5_src = Path(F5_TTS_PATH).expanduser().resolve() / "src"
    if str(f5_src) not in sys.path:
        sys.path.insert(0, str(f5_src))

from f5_tts.api import F5TTS  # noqa: E402
from f5_tts.infer.utils_infer import preprocess_ref_audio_text  # noqa: E402


OUTPUTS_DIR.mkdir(exist_ok=True)
SPECTROGRAMS_DIR.mkdir(exist_ok=True)

TEXT_EXAMPLES = [
    ("That's one small step for man, one giant leap for mankind. - Neil Armstrong", "That's one small step for man, one giant leap for mankind."),
    ("\"Toto, I've a feeling we're not in Kansas anymore.\" - The Wizard of Oz (1939)", "\"Toto, I've a feeling we're not in Kansas anymore.\""),
    ("We choose to go to the moon... - JFK", "We choose to go to the moon. We chose to go to the moon. We choose to go to the moon in this decade and do the other things not because they are easy, but because they are hard."),
    ("\"Here's looking at you, kid.\" - Casablanca (1942)", "\"Here's looking at you, kid.\""),
    ("\"May the Force be with you.\" - Star Wars (1977)", "\"May the Force be with you.\""),
    ("\"Fasten your seatbelts. It's going to be a bumpy night.\" - All About Eve (1950)", "\"Fasten your seatbelts. It's going to be a bumpy night.\""),
    ("\"A census taker once tried to test me...\" - The Silence of the Lambs (1991)", "\"A census taker once tried to test me. I ate his liver with some fava beans and a nice Chianti.\""),
    ("\"One morning I shot an elephant in my pajamas...\" - Animal Crackers (1930)", "\"One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't know.\""),
]
REFERENCE_PROMPT = "The quick brown fox jumps over the lazy dog."


def get_device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch, "xpu", None) and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_example_text(example_text: str) -> str:
    for label, value in TEXT_EXAMPLES:
        if example_text == label:
            return value
    return example_text or ""


def use_ref_text_mode(use_asr: bool):
    return gr.update(interactive=not use_asr)


def make_temp_file(suffix: str) -> str:
    handle = tempfile.NamedTemporaryFile(dir=OUTPUTS_DIR, suffix=suffix, delete=False)
    path = handle.name
    handle.close()
    return path


def make_temp_spectrogram_file() -> str:
    handle = tempfile.NamedTemporaryFile(dir=SPECTROGRAMS_DIR, suffix=".png", delete=False)
    path = handle.name
    handle.close()
    return path


def create_spectrogram(audio_path: str, title: str) -> str:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    n_fft = 1024
    hop_length = 256
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)(waveform)
    spectrogram_db = 10 * torch.log10(spectrogram + 1e-10)
    time_axis = np.arange(spectrogram_db.shape[-1]) * hop_length / sample_rate
    freq_axis = np.linspace(0, sample_rate / 2, spectrogram_db.shape[-2])

    output_path = make_temp_spectrogram_file()
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        spectrogram_db.squeeze(0).cpu().numpy(),
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


f5tts: F5TTS | None = None


def load_f5tts() -> F5TTS:
    global f5tts
    if f5tts is None:
        f5tts = F5TTS(device=get_device_label().split(" ", 1)[0])
    return f5tts


def clone_with_f5(
    reference_audio: str,
    reference_text: str,
    use_asr: bool,
    trim_generated_silence: bool,
    gen_text: str,
) -> tuple[str, str, str, str]:
    if not reference_audio:
        raise gr.Error("Upload or record a reference audio clip first.")
    if not gen_text or not gen_text.strip():
        raise gr.Error("Enter text to synthesize.")

    model = load_f5tts()

    ref_text = reference_text.strip()
    if not use_asr and not ref_text:
        raise gr.Error("Provide the reference transcript or enable ASR.")

    processed_ref_audio = reference_audio
    detected_ref_text = ref_text
    if use_asr:
        processed_ref_audio, detected_ref_text = preprocess_ref_audio_text(reference_audio, "")
    else:
        processed_ref_audio, detected_ref_text = preprocess_ref_audio_text(reference_audio, ref_text)
    reference_spectrogram_path = create_spectrogram(processed_ref_audio, "Reference Audio Spectrogram")

    wav_path = make_temp_file(".wav")

    model.infer(
        ref_file=processed_ref_audio,
        ref_text=detected_ref_text,
        gen_text=gen_text.strip(),
        remove_silence=trim_generated_silence,
        file_wave=wav_path,
    )
    generated_spectrogram_path = create_spectrogram(wav_path, "Generated Audio Spectrogram")

    return wav_path, reference_spectrogram_path, generated_spectrogram_path, detected_ref_text


def build_app() -> gr.Blocks:
    with gr.Blocks(title="F5-TTS Voice Cloning Demo") as demo:
        gr.Markdown(
            """
            # F5-TTS Voice Cloning Demo
            Upload or record a short reference clip, optionally auto-transcribe it, then synthesize new text with F5-TTS.
            """
        )
        gr.Markdown(f"Device: `{get_device_label()}`")
        gr.Markdown(f"Reference reading prompt: `{REFERENCE_PROMPT}`")

        with gr.Row():
            reference_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Voice",
            )
            output_audio = gr.Audio(
                type="filepath",
                label="Generated Audio",
            )

        with gr.Row():
            reference_spectrogram = gr.Image(
                type="filepath",
                label="Reference Spectrogram",
            )
            spectrogram = gr.Image(
                type="filepath",
                label="Generated Spectrogram",
            )

        use_asr = gr.Checkbox(
            value=True,
            label="Auto-Transcribe Reference",
        )
        trim_generated_silence = gr.Checkbox(
            value=True,
            label="Trim Generated Silence",
        )
        reference_text = gr.Textbox(
            label="Reference Transcript",
            lines=3,
            placeholder="Leave empty if Auto-Transcribe Reference is enabled.",
            interactive=False,
        )
        detected_ref_text = gr.Textbox(
            label="Detected Reference Transcript",
            lines=3,
            interactive=False,
        )
        gen_text = gr.Textbox(
            label="Text",
            lines=4,
            value=TEXT_EXAMPLES[0][1],
        )
        text_examples = gr.Dropdown(
            choices=[label for label, _ in TEXT_EXAMPLES],
            value=TEXT_EXAMPLES[0][0],
            label="Preset Text Examples",
            allow_custom_value=False,
        )
        generate = gr.Button("Generate", variant="primary")

        use_asr.change(
            fn=use_ref_text_mode,
            inputs=use_asr,
            outputs=reference_text,
        )
        text_examples.change(
            fn=use_example_text,
            inputs=text_examples,
            outputs=gen_text,
        )

        generate.click(
            fn=clone_with_f5,
            inputs=[reference_audio, reference_text, use_asr, trim_generated_silence, gen_text],
            outputs=[output_audio, reference_spectrogram, spectrogram, detected_ref_text],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    share = os.environ.get("SHARE", "").lower() in {"1", "true", "yes"}
    build_app().launch(server_name="0.0.0.0", server_port=port, share=share)
