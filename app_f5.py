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

COMPACT_AUDIO_CSS = """
.compact-audio {
    min-height: 82px !important;
}
.compact-audio .audio-container {
    height: 58px !important;
    min-height: 58px !important;
}
.compact-audio .component-wrapper {
    padding: 2px 8px !important;
}
"""


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
    ax.set_ylim(0, min(8000, sample_rate / 2))
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


def delete_reference_files(reference_audio: str, processed_ref_audio: str | None) -> None:
    """Delete Gradio's recording and F5-TTS's processed temp copy."""
    paths: set[Path] = set()
    if processed_ref_audio:
        processed_path = Path(processed_ref_audio).resolve()
        if processed_path != Path(reference_audio).resolve():
            paths.add(processed_path)

    gradio_temp_root = Path(
        os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "gradio")
    ).resolve()
    reference_path = Path(reference_audio).resolve()
    if reference_path.is_relative_to(gradio_temp_root):
        paths.add(reference_path)

    failures: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
            if path.exists():
                failures.append(f"{path}: file still exists")
        except OSError as error:
            failures.append(f"{path}: {error}")

    if failures:
        raise RuntimeError("Could not delete reference audio:\n" + "\n".join(failures))


def clone_with_f5(
    reference_audio: str,
    trim_generated_silence: bool,
    gen_text: str,
) -> tuple[None, str, str, str]:
    if not reference_audio:
        raise gr.Error("Record a reference audio clip first.")
    if not gen_text or not gen_text.strip():
        raise gr.Error("Enter text to synthesize.")

    processed_ref_audio: str | None = None
    try:
        model = load_f5tts()
        processed_ref_audio, detected_ref_text = preprocess_ref_audio_text(reference_audio, "")
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

        # Returning None clears the deleted reference recording from the UI.
        return None, wav_path, reference_spectrogram_path, generated_spectrogram_path
    finally:
        delete_reference_files(reference_audio, processed_ref_audio)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="F5-TTS Voice Cloning Demo", css=COMPACT_AUDIO_CSS) as demo:
        gr.Markdown(
            """
            # F5-TTS Voice Cloning Demo
            **Please read aloud:** “The quick brown fox jumps over the lazy dog.”
            """
        )

        with gr.Row():
            reference_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Reference Voice",
                interactive=True,
                waveform_options={"show_recording_waveform": False},
                elem_classes="compact-audio",
            )
            output_audio = gr.Audio(
                type="filepath",
                label="Generated Audio",
                interactive=False,
                waveform_options={"show_recording_waveform": False},
                elem_classes="compact-audio",
            )

        with gr.Row():
            gen_text = gr.Textbox(
                label="Text to Generate",
                lines=2,
                value=TEXT_EXAMPLES[0][1],
                scale=2,
            )
            text_examples = gr.Dropdown(
                choices=[label for label, _ in TEXT_EXAMPLES],
                value=TEXT_EXAMPLES[0][0],
                label="Preset Text Examples",
                allow_custom_value=False,
                scale=1,
            )

        with gr.Row():
            trim_generated_silence = gr.Checkbox(
                value=True,
                label="Trim Generated Silence",
            )
            generate = gr.Button("Generate", variant="primary")

        with gr.Row():
            reference_spectrogram = gr.Image(
                type="filepath",
                label="Reference Spectrogram (0–8 kHz)",
                height=240,
                show_download_button=False,
                show_fullscreen_button=False,
            )
            spectrogram = gr.Image(
                type="filepath",
                label="Generated Spectrogram (0–8 kHz)",
                height=240,
                show_download_button=False,
                show_fullscreen_button=False,
            )

        text_examples.change(
            fn=use_example_text,
            inputs=text_examples,
            outputs=gen_text,
        )

        generate.click(
            fn=clone_with_f5,
            inputs=[reference_audio, trim_generated_silence, gen_text],
            outputs=[reference_audio, output_audio, reference_spectrogram, spectrogram],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    share = os.environ.get("SHARE", "").lower() in {"1", "true", "yes"}
    build_app().launch(server_name="0.0.0.0", server_port=port, share=share)
