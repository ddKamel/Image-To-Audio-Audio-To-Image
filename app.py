"""
app.py

Streamlit UI for the Image-to-Sound / Sound-to-Image project.

Run with:
    streamlit run app.py

Three tabs:
- "Image -> Audio": turn an image into audio (image treated as spectrogram)
- "Audio -> Image": turn audio into an image (compute its spectrogram),
  optionally applying an audio effect first to see how it changes the image
- "Paint Spectrogram": draw a spectrogram by hand and hear it
"""

import os
import tempfile

import numpy as np
import librosa
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from core_pipeline import (
    image_to_audio,
    audio_to_image,
    save_audio,
    make_test_image_single_line,
    make_test_image_harmonic_stack,
    synthesize_sine_wave,
    note_name_to_frequency,
    apply_filter,
    apply_distortion,
)

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


def show_round_trip_comparison(original_image_path: str, reconstructed_audio_path: str, n_fft: int):
    """
    Computes the spectrogram of the just-reconstructed audio and shows it
    next to the original input image, so information loss/artifacts from
    Griffin-Lim + quantization + resizing become visible, not just audible.
    """
    round_trip_image_path = os.path.join(tmp_dir, "round_trip.png")
    audio_to_image(reconstructed_audio_path, round_trip_image_path, n_fft=n_fft, sr=SAMPLE_RATE)

    st.write("**Round-trip comparison** (original image vs. spectrogram of the audio you just heard):")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(make_display_thumbnail(original_image_path), caption="Original image")
    with col_b:
        st.image(make_display_thumbnail(round_trip_image_path), caption="Spectrogram of reconstructed audio")


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

    # Streamlit does not allow passing `value=` for a widget whose key is
    # already present in session_state (it doesn't know which should win).
    # So we only seed the initial value on the very first run for each key;
    # after that, the widget reads its value from session_state via `key`.
    slider_kwargs = {} if slider_key in st.session_state else {"value": nearest_preset}
    input_kwargs = {} if input_key in st.session_state else {"value": current}

    col_slider, col_input = st.columns([2, 1])
    with col_slider:
        st.select_slider(
            "Duration (log scale, quick pick)",
            options=_DURATION_PRESETS,
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
            **slider_kwargs,
        )
    with col_input:
        st.number_input(
            "Exact seconds",
            min_value=1.0,
            max_value=600.0,
            step=1.0,
            key=input_key,
            on_change=_on_input_change,
            **input_kwargs,
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
                ""
                "With the standard set Sample Rate of 22050 Hz an FFT size of 2048 is optimal "
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
                " "
                "Example: a small value makes dark image areas "
                "audible, giving a fuller/noisier sound; a large value"
                "silences dark areas, making the sound cleaner and more selective."
            ),
            key="i2a_db",
        )
        gl_iterations = st.slider(
            "Griffin-Lim iterations",
            4, 64, 32, step=4,
            help=(
                "How many iterations are used to estimate the missing phase "
                "information using Griffin-Lim."
                " "
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

            show_round_trip_comparison(image_path, out_audio_path, n_fft)
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
        st.write("Original audio:")
        st.audio(audio_path)

        st.subheader("Effects (optional)")
        st.caption("Apply an effect to the audio and see how it changes the spectrogram image.")

        apply_filter_enabled = st.checkbox("Apply filter", key="a2i_filter_enabled")
        if apply_filter_enabled:
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                filter_type = st.radio(
                    "Filter type", ["lowpass", "highpass"], horizontal=True, key="a2i_filter_type"
                )
            with fcol2:
                cutoff_hz = st.slider(
                    "Cutoff frequency (Hz)", 50, 10000, 2000, step=50, key="a2i_filter_cutoff",
                    help=(
                        "Lowpass keeps frequencies below this and removes everything "
                        "above - the image should look cut off/dark above this "
                        "frequency's row. Highpass does the opposite."
                    ),
                )

        apply_distortion_enabled = st.checkbox("Apply distortion", key="a2i_distortion_enabled")
        if apply_distortion_enabled:
            drive = st.slider(
                "Drive", 1.0, 20.0, 5.0, step=0.5, key="a2i_distortion_drive",
                help=(
                    "How hard the signal is pushed into the clipping curve. "
                    "Higher values add more new harmonic content - expect new "
                    "energy to appear above the original frequencies in the image."
                ),
            )

        effects_active = apply_filter_enabled or apply_distortion_enabled

        if st.button("Generate Image", type="primary", key="a2i_generate"):
            with st.spinner("Computing spectrogram..."):
                out_image_path = os.path.join(tmp_dir, "spectrogram.png")
                audio_to_image(audio_path, out_image_path, n_fft=n_fft2, db_range=db_range2, sr=SAMPLE_RATE)

                processed_audio_path = None
                processed_image_path = None
                if effects_active:
                    processed_audio, sr_loaded = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    if apply_filter_enabled:
                        processed_audio = apply_filter(
                            processed_audio, SAMPLE_RATE, filter_type=filter_type, cutoff_hz=cutoff_hz
                        )
                    if apply_distortion_enabled:
                        processed_audio = apply_distortion(processed_audio, drive=drive)

                    processed_audio_path = os.path.join(tmp_dir, "processed_audio.wav")
                    save_audio(processed_audio, SAMPLE_RATE, processed_audio_path)

                    processed_image_path = os.path.join(tmp_dir, "processed_spectrogram.png")
                    audio_to_image(
                        processed_audio_path, processed_image_path, n_fft=n_fft2, db_range=db_range2, sr=SAMPLE_RATE
                    )

            st.success("Done")

            if effects_active:
                st.write("With effects applied:")
                st.audio(processed_audio_path)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(make_display_thumbnail(out_image_path), caption="Original")
                with col_b:
                    st.image(make_display_thumbnail(processed_image_path), caption="With effects")
                with open(processed_image_path, "rb") as f:
                    st.download_button(
                        "Download full-resolution image with effects (.png)",
                        f,
                        file_name="spectrogram_with_effects.png",
                        key="a2i_download_processed",
                    )
            else:
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
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 250

if "paint_canvas_array" not in st.session_state:
    st.session_state.paint_canvas_array = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)


def _row_to_frequency(row: int, sr: int, canvas_height: int = CANVAS_HEIGHT) -> float:
    """Approximate frequency represented by a given canvas row (0 = top =
    high frequency, canvas_height = bottom = 0 Hz), matching the mapping
    image_to_audio() uses after flipping the image vertically."""
    return (1.0 - row / canvas_height) * (sr / 2.0)


def _frequency_to_row(freq_hz: float, sr: int, canvas_height: int = CANVAS_HEIGHT) -> int:
    row = canvas_height * (1.0 - freq_hz / (sr / 2.0))
    return int(max(0, min(canvas_height - 1, row)))


def _draw_plain_stroke(arr: np.ndarray, x1, y1, x2, y2, stroke_width: int, brightness: int):
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)
    draw.line([(x1, y1), (x2, y2)], fill=brightness, width=stroke_width)
    if brightness > 0:
        # round the ends of the stroke so single clicks (x1==x2, y1==y2) show up
        r = stroke_width / 2
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=brightness)
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=brightness)
    return np.array(img)


def _draw_harmonic_stack_stroke(
        arr: np.ndarray, x1, y1, x2, y2, stroke_width: int, sr: int,
        n_harmonics: int = 5, decay: float = 0.6,
):
    """Draws the fundamental stroke plus fainter strokes at integer multiples
    of the implied frequency above it - approximates an instrument-like
    harmonic series instead of a pure tone."""
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)

    # Sample a few points along the stroke, compute the fundamental frequency
    # at each, then draw the harmonic rows for each sampled x position.
    n_samples = max(2, int(abs(x2 - x1)) + 1)
    for t in np.linspace(0, 1, n_samples):
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        f0 = _row_to_frequency(y, sr)
        if f0 <= 0:
            continue
        for k in range(1, n_harmonics + 1):
            harmonic_freq = f0 * k
            if harmonic_freq >= sr / 2:
                break
            row_k = _frequency_to_row(harmonic_freq, sr)
            brightness_k = int(255 * (decay ** (k - 1)))
            half_w = max(1, stroke_width // 2)
            draw.line(
                [(x, row_k - half_w / 2), (x, row_k + half_w / 2)],
                fill=brightness_k, width=max(1, stroke_width // 2),
            )
    return np.array(img)


with tab_paint:
    st.caption(
        "Draw directly on a blank spectrogram: horizontal axis = time, "
        "vertical axis = frequency (high at the top, low at the bottom), "
        "brightness = loudness."
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
    tool_col, shape_col, width_col = st.columns(3)
    with tool_col:
        tool = st.radio("Tool", ["Brush", "Eraser"], key="paint_tool")
    with shape_col:
        pen_shape = st.radio(
            "Pen shape", ["Plain line", "Harmonic stack"], key="paint_pen_shape",
            help=(
                "Plain line: draws exactly what you drag - good for pure "
                "tones and sweeps. Harmonic stack: also adds fainter lines "
                "at integer multiples of the frequency you draw, so it "
                "sounds more like an instrument playing a note than a pure "
                "sine tone."
            ),
        )
    with width_col:
        stroke_width = st.slider("Stroke width", 1, 30, 8, key="paint_stroke_width")

    st.caption(
        "Click and drag on the canvas below to draw a stroke. Each drag adds "
        "one stroke; keep dragging to build up the picture."
    )

    canvas_image = Image.fromarray(st.session_state.paint_canvas_array, mode="L")
    coords = streamlit_image_coordinates(
        canvas_image,
        key="paint_canvas_widget",
        click_and_drag=True,
        width=CANVAS_WIDTH,
    )

    if coords is not None:
        # Scale from the coordinates reported by the browser (which may
        # differ from CANVAS_WIDTH/HEIGHT if the image was displayed at a
        # different size) back to the underlying array's pixel space.
        scale_x = CANVAS_WIDTH / max(coords.get("width", CANVAS_WIDTH), 1)
        scale_y = CANVAS_HEIGHT / max(coords.get("height", CANVAS_HEIGHT), 1)
        x1 = coords["x1"] * scale_x
        y1 = coords["y1"] * scale_y
        x2 = coords["x2"] * scale_x
        y2 = coords["y2"] * scale_y

        stroke_id = coords.get("unix_time")
        if st.session_state.get("paint_last_stroke") != stroke_id:
            st.session_state.paint_last_stroke = stroke_id

            if tool == "Eraser":
                st.session_state.paint_canvas_array = _draw_plain_stroke(
                    st.session_state.paint_canvas_array, x1, y1, x2, y2, stroke_width, brightness=0
                )
            elif pen_shape == "Plain line":
                st.session_state.paint_canvas_array = _draw_plain_stroke(
                    st.session_state.paint_canvas_array, x1, y1, x2, y2, stroke_width, brightness=255
                )
            else:
                st.session_state.paint_canvas_array = _draw_harmonic_stack_stroke(
                    st.session_state.paint_canvas_array, x1, y1, x2, y2, stroke_width, sr=SAMPLE_RATE
                )
            st.rerun()

    if st.button("Clear canvas", key="paint_clear"):
        st.session_state.paint_canvas_array = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)
        st.rerun()

    if st.button("Generate Audio from Painting", type="primary", key="paint_generate"):
        if not np.any(st.session_state.paint_canvas_array):
            st.warning("The canvas is empty - draw something first.")
        else:
            with st.spinner("Running Griffin-Lim reconstruction..."):
                painted_image_path = os.path.join(tmp_dir, "painted_spectrogram.png")
                Image.fromarray(st.session_state.paint_canvas_array, mode="L").save(painted_image_path)

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

            show_round_trip_comparison(painted_image_path, out_audio_path, paint_n_fft)
