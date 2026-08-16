"""
app.py

Streamlit UI for the Image-to-Sound / Sound-to-Image project.

Run with:
    streamlit run app.py

Two tabs keep the two directions of the pipeline separate:
- "Image -> Audio": turn an image into audio (image treated as spectrogram)
- "Audio -> Image": turn audio into an image (compute its spectrogram)
"""

import os
import tempfile

import streamlit as st
from PIL import Image

from core_pipeline import (
    image_to_audio,
    audio_to_image,
    save_audio,
    make_test_image_single_line,
    make_test_image_harmonic_stack,
    synthesize_sine_wave,
    note_name_to_frequency,
)

import numpy as np
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Image to Sound", layout="centered")

st.title("Image to Sound")
st.caption(
    "An image is treated directly as a spectrogram: brightness <-> magnitude, "
    "phase is estimated with Griffin-Lim, audio is reconstructed via inverse STFT."
)

tmp_dir = tempfile.mkdtemp(prefix="img2sound_")
SAMPLE_RATE = 22050

# Every preview image is resized to this size before being displayed, so the
# UI stays consistent regardless of the actual resolution used internally
# (e.g. a spectrogram image can have 1000+ rows depending on n_fft).
# This matches the default size of the built-in test images.
DISPLAY_SIZE = (200, 200)


def make_display_thumbnail(src_path: str) -> str:
    """Creates a small preview copy of an image for display purposes only.
    The original file (used for actual audio processing) is left untouched."""
    thumb_path = os.path.join(tmp_dir, "thumb_" + os.path.basename(src_path))
    img = Image.open(src_path).convert("L")
    img = img.resize(DISPLAY_SIZE)
    img.save(thumb_path)
    return thumb_path


# Log-spaced preset points (in seconds) for the quick-pick duration slider.
# Covers 1 second up to 10 minutes (600 seconds).
_DURATION_PRESETS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600]


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}m {rest}s" if rest else f"{minutes}m"


def duration_control(key_prefix: str, default_sec: float = 3.0) -> float:
    """
    Renders a duration control that offers both a log-scaled quick-pick
    slider (better for short, musically relevant durations) and a plain
    number input (for typing an exact value, e.g. to match a specific
    audio file length up to 10 minutes). Both stay in sync.
    """
    master_key = f"{key_prefix}_duration_sec"
    slider_key = f"{key_prefix}_duration_slider"
    input_key = f"{key_prefix}_duration_input"

    if master_key not in st.session_state:
        st.session_state[master_key] = default_sec

    def _on_slider_change():
        val = float(st.session_state[slider_key])
        st.session_state[master_key] = val
        st.session_state[input_key] = val

    def _on_input_change():
        val = float(st.session_state[input_key])
        val = max(1.0, min(600.0, val))
        st.session_state[master_key] = val
        st.session_state[slider_key] = min(
            _DURATION_PRESETS, key=lambda p: abs(p - val)
        )

    current = st.session_state[master_key]
    nearest_preset = min(_DURATION_PRESETS, key=lambda p: abs(p - current))

    col_slider, col_input = st.columns([2, 1])
    with col_slider:
        st.select_slider(
            "Duration (log scale, quick pick)",
            options=_DURATION_PRESETS,
            value=nearest_preset,
            format_func=_format_duration,
            key=slider_key,
            on_change=_on_slider_change,
            help=(
                "Quick-pick duration on a logarithmic scale, so short "
                "(sub-second-to-second) and long (multi-minute) durations "
                "are both easy to reach. Type an exact value on the right "
                "if you need a specific number of seconds, e.g. to match "
                "an uploaded song's length."
            ),
        )
    with col_input:
        st.number_input(
            "Exact seconds",
            min_value=1.0,
            max_value=600.0,
            step=1.0,
            value=float(st.session_state[input_key]) if input_key in st.session_state else current,
            key=input_key,
            on_change=_on_input_change,
        )

    return st.session_state[master_key]


tab_img_to_audio, tab_audio_to_img, tab_paint = st.tabs(
    ["Image \u2192 Audio", "Audio \u2192 Image", "Paint Spectrogram"]
)


# =============================================================================
# TAB 1: Image -> Audio
# =============================================================================
with tab_img_to_audio:
    st.subheader("Parameters")
    duration_sec = duration_control("i2a", default_sec=3.0)
    col1, col2 = st.columns(2)
    with col1:
        n_fft = st.select_slider(
            "FFT size (frequency resolution)",
            options=[512, 1024, 2048, 4096], value=2048,
            help=(
                "Number of frequency bins the image is stretched to vertically. "
                "Higher = finer frequency detail but coarser timing per frame; "
                "lower = the opposite. "
                "Example: a thin horizontal line becomes a purer, narrower tone "
                "at n_fft=4096 than at n_fft=512, where nearby frequencies blur "
                "together."
            ),
            key="i2a_nfft",
        )
    with col2:
        db_range = st.slider(
            "Dynamic range (dB)",
            20.0, 120.0, 80.0, step=5.0,
            help=(
                "Maps pixel brightness to loudness: black = -db_range dB (quiet), "
                "white = 0 dB (loudest). "
                "Example: a small value (20 dB) makes even dark image areas "
                "audible, giving a fuller/noisier sound; a large value (120 dB) "
                "silences dark areas, making the sound cleaner and more selective."
            ),
            key="i2a_db",
        )
        gl_iterations = st.slider(
            "Griffin-Lim iterations",
            4, 64, 32, step=4,
            help=(
                "How many iterations are used to estimate the missing phase "
                "information (the image only encodes magnitude, not phase). "
                "Example: 4 iterations reconstruct quickly but sound metallic/"
                "noisy; 32-64 iterations sound cleaner but take longer to compute."
            ),
            key="i2a_gl",
        )

    source = st.radio(
        "Image source",
        [
            "Upload my own image",
            "Use a test image (single tone)",
            "Use a test image (harmonics)",
        ],
        horizontal=True,
        key="i2a_source",
    )

    image_path = None

    if source == "Upload my own image":
        uploaded_file = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg", "bmp"], key="i2a_upload"
        )
        if uploaded_file is not None:
            image_path = os.path.join(
                tmp_dir, "uploaded" + os.path.splitext(uploaded_file.name)[1]
            )
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    elif source == "Use a test image (single tone)":
        image_path = os.path.join(tmp_dir, "test_single_line.png")
        make_test_image_single_line(image_path)
    else:
        image_path = os.path.join(tmp_dir, "test_harmonics.png")
        make_test_image_harmonic_stack(image_path)

    if image_path is not None:
        st.image(
            make_display_thumbnail(image_path),
            caption="Input image (interpreted as spectrogram)",
        )

        if st.button("Generate Audio", type="primary", key="i2a_generate"):
            with st.spinner("Running Griffin-Lim reconstruction..."):
                audio, out_sr = image_to_audio(
                    image_path,
                    n_fft=n_fft,
                    sr=SAMPLE_RATE,
                    duration_sec=duration_sec,
                    db_range=db_range,
                    gl_iterations=gl_iterations,
                )
                out_audio_path = os.path.join(tmp_dir, "output.wav")
                save_audio(audio, out_sr, out_audio_path)

            st.success("Done")
            st.audio(out_audio_path, format="audio/wav")
            with open(out_audio_path, "rb") as f:
                st.download_button(
                    "Download .wav", f, file_name="reconstructed.wav", key="i2a_download"
                )
    else:
        st.info("Please upload an image or select a test image to get started.")


# =============================================================================
# TAB 2: Audio -> Image
# =============================================================================
with tab_audio_to_img:
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_fft2 = st.select_slider(
            "FFT size (frequency resolution)",
            options=[512, 1024, 2048, 4096], value=2048,
            help=(
                "Determines how many frequency bins (rows) the resulting "
                "spectrogram image has, and how precisely close frequencies "
                "can be told apart. "
                "Example: a pure tone looks like a single sharp horizontal "
                "line at n_fft=4096, but a slightly blurrier band at n_fft=512."
            ),
            key="a2i_nfft",
        )
    with col2:
        db_range2 = st.slider(
            "Dynamic range (dB)",
            20.0, 120.0, 80.0, step=5.0,
            help=(
                "How much of the quiet part of the sound still shows up as "
                "visible (non-black) pixels in the image. "
                "Example: a small value (20 dB) makes quiet background noise "
                "visible as faint gray; a large value (120 dB) shows only the "
                "loudest parts, rest stays black."
            ),
            key="a2i_db",
        )

    source2 = st.radio(
        "Audio source",
        ["Play a specific note (piano)", "Upload an audio file"],
        horizontal=True,
        key="a2i_source",
    )

    audio_path = None

    if source2 == "Play a specific note (piano)":
        st.write("Pick a note:")

        NATURAL_NOTES = ["C", "D", "E", "F", "G", "A", "B"]
        # sharp key, placed above the gap between the correct naturals
        # (no entry = no sharp after that natural, e.g. no E# / B#)
        SHARPS_AFTER = {"C": "C#", "D": "D#", "F": "F#", "G": "G#", "A": "A#"}

        if "a2i_selected_note" not in st.session_state:
            st.session_state.a2i_selected_note = "G#"

        # Row 1: black keys
        black_cols = st.columns(len(NATURAL_NOTES))
        for i, natural in enumerate(NATURAL_NOTES):
            sharp = SHARPS_AFTER.get(natural)
            with black_cols[i]:
                if sharp is not None:
                    if st.button(sharp, key=f"a2i_key_{sharp}", use_container_width=True):
                        st.session_state.a2i_selected_note = sharp

        # Row 2: white keys
        white_cols = st.columns(len(NATURAL_NOTES))
        for i, natural in enumerate(NATURAL_NOTES):
            with white_cols[i]:
                if st.button(natural, key=f"a2i_key_{natural}", use_container_width=True):
                    st.session_state.a2i_selected_note = natural

        octave = st.slider("Octave", 0, 7, 2, key="a2i_octave")
        note_duration_sec = st.slider(
            "Duration (seconds)", 1.0, 5.0, 2.0, step=0.5, key="a2i_note_duration"
        )
        full_note = f"{st.session_state.a2i_selected_note}{octave}"

        try:
            freq = note_name_to_frequency(full_note)
            nyquist = SAMPLE_RATE / 2.0
            if freq >= nyquist:
                st.error(
                    f"{full_note} ({freq:.2f} Hz) is above the Nyquist frequency "
                    f"({nyquist:.2f} Hz) at this sample rate. Pick a lower octave."
                )
            else:
                st.write(f"Selected note: **{full_note}** ({freq:.2f} Hz)")
                audio = synthesize_sine_wave(freq, duration_sec=note_duration_sec, sr=SAMPLE_RATE)
                audio_path = os.path.join(tmp_dir, "note_audio.wav")
                save_audio(audio, SAMPLE_RATE, audio_path)
        except Exception as e:
            st.error(f"Could not parse note: {e}")

    else:
        st.caption("Note: .mp3 support depends on the audio backend installed on your system.")
        uploaded_audio = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3"], key="a2i_upload"
        )
        if uploaded_audio is not None:
            audio_path = os.path.join(
                tmp_dir, "uploaded_audio" + os.path.splitext(uploaded_audio.name)[1]
            )
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

    if audio_path is not None:
        st.audio(audio_path)

        if st.button("Generate Image", type="primary", key="a2i_generate"):
            with st.spinner("Computing spectrogram..."):
                out_image_path = os.path.join(tmp_dir, "spectrogram.png")
                audio_to_image(audio_path, out_image_path, n_fft=n_fft2, db_range=db_range2, sr=SAMPLE_RATE)

            st.success("Done")
            st.image(
                make_display_thumbnail(out_image_path),
                caption="Resulting spectrogram image",
            )
            with open(out_image_path, "rb") as f:
                st.download_button(
                    "Download full-resolution image (.png)",
                    f,
                    file_name="spectrogram.png",
                    key="a2i_download",
                )
    else:
        st.info("Please select a note or upload an audio file to get started.")


# =============================================================================
# TAB 3: Paint Spectrogram
# =============================================================================
with tab_paint:
    st.caption(
        "Draw directly on a blank spectrogram: horizontal axis = time, "
        "vertical axis = frequency (high at the top, low at the bottom), "
        "brightness = loudness. White strokes = sound, black erases it."
    )

    st.subheader("Parameters")
    paint_duration_sec = duration_control("paint", default_sec=3.0)
    col1, col2 = st.columns(2)
    with col1:
        paint_n_fft = st.select_slider(
            "FFT size (frequency resolution)",
            options=[512, 1024, 2048, 4096], value=2048,
            key="paint_nfft",
        )
    with col2:
        paint_db_range = st.slider(
            "Dynamic range (dB)", 20.0, 120.0, 80.0, step=5.0, key="paint_db"
        )
        paint_gl_iterations = st.slider(
            "Griffin-Lim iterations", 4, 64, 32, step=4, key="paint_gl"
        )

    st.subheader("Canvas")
    tool_col, width_col = st.columns(2)
    with tool_col:
        tool = st.radio("Tool", ["Brush", "Eraser"], horizontal=True, key="paint_tool")
    with width_col:
        stroke_width = st.slider("Stroke width", 1, 30, 8, key="paint_stroke_width")

    # Brush paints white (= loud), eraser paints black (= silent) - simplest
    # way to support erasing without needing a separate transparency layer.
    stroke_color = "#FFFFFF" if tool == "Brush" else "#000000"

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        height=300,
        width=600,
        drawing_mode="freedraw",
        display_toolbar=True,
        key="paint_canvas",
    )

    if st.button("Generate Audio from Painting", type="primary", key="paint_generate"):
        if canvas_result.image_data is None or not np.any(canvas_result.image_data[:, :, :3]):
            st.warning("The canvas is empty - draw something first.")
        else:
            with st.spinner("Running Griffin-Lim reconstruction..."):
                # Canvas gives RGBA; drop alpha and convert to grayscale.
                rgb = canvas_result.image_data[:, :, :3].astype(np.uint8)
                gray = Image.fromarray(rgb, mode="RGB").convert("L")
                painted_image_path = os.path.join(tmp_dir, "painted_spectrogram.png")
                gray.save(painted_image_path)

                audio, out_sr = image_to_audio(
                    painted_image_path,
                    n_fft=paint_n_fft,
                    sr=SAMPLE_RATE,
                    duration_sec=paint_duration_sec,
                    db_range=paint_db_range,
                    gl_iterations=paint_gl_iterations,
                )
                out_audio_path = os.path.join(tmp_dir, "painted_output.wav")
                save_audio(audio, out_sr, out_audio_path)

            st.success("Done")
            st.audio(out_audio_path, format="audio/wav")
            with open(out_audio_path, "rb") as f:
                st.download_button(
                    "Download .wav",
                    f,
                    file_name="painted_reconstruction.wav",
                    key="paint_download",
                )
