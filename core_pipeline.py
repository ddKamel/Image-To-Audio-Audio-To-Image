"""
core_pipeline.py

Core of the Image-to-Sound / Sound-to-Image project.

Idea (Functionality 3 - STFT Resynthesis):
An image is NOT analyzed via FFT. Instead, it is interpreted directly as a
spectrogram: columns = time, rows = frequency bins, brightness = magnitude.
From this magnitude, a plausible phase is estimated via Griffin-Lim, and an
audio signal is reconstructed from it via inverse STFT.

Reverse direction (Sound-to-Image): Audio -> STFT -> |magnitude| -> log-scaled
-> saved as a grayscale image.

Dependencies: numpy, librosa, soundfile, pillow
Installation: pip install numpy librosa soundfile pillow
"""

import numpy as np
from PIL import Image
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Image -> Audio
# ---------------------------------------------------------------------------

def image_to_audio(
    image_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    sr: int = 22050,
    duration_sec: float = 4.0,
    db_range: float = 80.0,
    gl_iterations: int = 32,
) -> np.ndarray:
    """
    Interprets an image directly as a spectrogram and reconstructs an audio
    signal from it via Griffin-Lim.

    Parameters
    ----------
    image_path : path to the input image
    n_fft      : FFT window size -> determines number of frequency bins (n_fft//2 + 1)
    hop_length : hop size between STFT frames
    sr         : target sample rate
    duration_sec : desired audio duration -> determines number of time frames (columns)
    db_range   : dynamic range in dB that brightness is mapped onto
                 (0 = silence, db_range = loudest level)
    gl_iterations : number of Griffin-Lim iterations (more = cleaner phase,
                    but slower)

    Returns
    -------
    audio : np.ndarray, float32, reconstructed time-domain signal
    """
    # 1) Load as grayscale
    img = Image.open(image_path).convert("L")

    # 2) Compute target size: rows = frequency bins, columns = time frames
    n_freq_bins = n_fft // 2 + 1
    n_time_frames = max(1, int(np.round(duration_sec * sr / hop_length)))
    img = img.resize((n_time_frames, n_freq_bins))

    # 3) Convert to array, flip vertically
    #    (image row 0 is at the top -> but should correspond to high frequency,
    #     "bottom" row = low frequency)
    magnitude_img = np.asarray(img, dtype=np.float32)
    magnitude_img = np.flipud(magnitude_img)

    # 4) Brightness (0-255) -> dB scale -> linear magnitude
    #    0   -> -db_range dB (quiet/silent)
    #    255 -> 0 dB (loudest level of this spectrum)
    normalized = magnitude_img / 255.0
    db = (normalized * db_range) - db_range          # range: [-db_range, 0]
    magnitude = librosa.db_to_amplitude(db)

    # 5) Griffin-Lim: estimate phase, apply inverse STFT
    audio = librosa.griffinlim(
        magnitude,
        n_iter=gl_iterations,
        hop_length=hop_length,
        win_length=n_fft,
    )

    # 6) Peak-normalize. Long/heavily downsampled images especially can end
    #    up with very low output amplitude after Griffin-Lim, to the point
    #    of being barely audible even though the signal is not literally
    #    silent. This brings the loudest sample up to a safe target peak.
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak * 0.95

    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, sr: int, out_path: str) -> None:
    sf.write(out_path, audio, sr)


# ---------------------------------------------------------------------------
# Audio -> Image
# ---------------------------------------------------------------------------

def audio_to_image(
    audio_path: str,
    out_image_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    db_range: float = 80.0,
    sr: int = 22050,
) -> None:
    """
    Computes the spectrogram of an audio signal and saves it as a grayscale
    image (columns = time, rows = frequency, bottom = low frequency).

    IMPORTANT: the audio is resampled to the given sr (default 22050) rather
    than kept at its native rate. This must match the sr used later in
    image_to_audio() when reconstructing - otherwise the row-to-frequency
    mapping differs between encoding and decoding, and the reconstructed
    audio comes out pitch-shifted (e.g. loading a 44.1 kHz file at its
    native rate here, then reconstructing assuming 22.05 kHz, halves every
    frequency - one octave down).
    """
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    db = librosa.amplitude_to_db(magnitude, ref=np.max)   # range: [-inf, 0]
    db = np.clip(db, -db_range, 0)

    normalized = (db + db_range) / db_range                # range: [0, 1]
    img_array = np.flipud(normalized * 255.0).astype(np.uint8)

    Image.fromarray(img_array, mode="L").save(out_image_path)


# ---------------------------------------------------------------------------
# Sanity check: synthetic test images
# ---------------------------------------------------------------------------

def make_test_image_single_line(
    path: str,
    width: int = 200,
    height: int = 200,
    line_row_frac: float = 0.3,
    line_thickness: int = 3,
) -> None:
    """
    Creates a test image with a single horizontal white line on a black
    background. Should sound like a pure sine tone after the pipeline
    (pitch depends on line_row_frac).
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    row = int(height * line_row_frac)
    row = max(0, min(height - 1, row))
    r0 = max(0, row - line_thickness // 2)
    r1 = min(height, r0 + line_thickness)
    arr[r0:r1, :] = 255
    Image.fromarray(arr, mode="L").save(path)


def make_test_image_harmonic_stack(
    path: str,
    width: int = 200,
    height: int = 200,
    fundamental_frac: float = 0.15,
    n_harmonics: int = 5,
) -> None:
    """
    Multiple horizontal lines at integer multiples of a fundamental
    frequency -> should sound like an instrument/chord with overtones
    instead of a pure sine tone.
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    for k in range(1, n_harmonics + 1):
        frac = fundamental_frac * k
        if frac >= 1.0:
            break
        row = int(height * frac)
        brightness = int(255 / k)  # higher harmonics quieter
        arr[row, :] = brightness
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# Note-to-frequency helper (for building a target pure tone image)
# ---------------------------------------------------------------------------

_NOTE_INDEX = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
    "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}


def note_name_to_frequency(note_name: str) -> float:
    """
    Converts a note name like "G#2" or "A4" into its frequency in Hz,
    using the standard convention A4 = 440 Hz.
    """
    note_name = note_name.strip().upper()
    # split into pitch class (letter + optional #) and octave number
    if len(note_name) >= 2 and note_name[1] == "#":
        pitch_class, octave_str = note_name[:2], note_name[2:]
    else:
        pitch_class, octave_str = note_name[:1], note_name[1:]

    octave = int(octave_str)
    note_idx = _NOTE_INDEX[pitch_class]

    midi_number = 12 * (octave + 1) + note_idx
    frequency = 440.0 * (2.0 ** ((midi_number - 69) / 12.0))
    return frequency


def make_test_image_pure_tone(
    path: str,
    note_name: str,
    sr: int = 22050,
    n_fft: int = 2048,
    width: int = 200,
    line_thickness: int = 1,
) -> float:
    """
    Creates a single-line test image tuned to reconstruct as close as
    possible to the given musical note (e.g. "G#2", "A4").

    Unlike make_test_image_single_line(), this computes the exact target
    frequency bin directly from n_fft and sets the image height to exactly
    n_fft // 2 + 1 (the number of frequency bins produced by that n_fft).
    This way there is no resizing/interpolation involved later in
    image_to_audio() as long as the SAME n_fft is passed there too -
    the line lands exactly on the intended bin.

    IMPORTANT: sr and n_fft here must match the values later used in
    image_to_audio(), otherwise the pitch will land on the wrong bin.

    Returns
    -------
    frequency : the target frequency in Hz, for reference/printing.
    """
    frequency = note_name_to_frequency(note_name)
    nyquist = sr / 2.0
    if frequency >= nyquist:
        raise ValueError(
            f"{note_name} ({frequency:.2f} Hz) is above the Nyquist frequency "
            f"({nyquist:.2f} Hz) for sr={sr}. Increase sr or pick a lower note."
        )

    # Exact frequency bin this note falls into, given n_fft and sr.
    n_freq_bins = n_fft // 2 + 1
    bin_index = int(round(frequency * n_fft / sr))
    bin_index = max(0, min(n_freq_bins - 1, bin_index))

    # image_to_audio() flips the image vertically before treating row 0 as
    # bin 0, so to land on `bin_index` after that flip, the line must be
    # drawn at this row (counted from the top, before the flip):
    row_from_top = n_freq_bins - 1 - bin_index

    # Height is set to exactly n_freq_bins so image_to_audio() does not
    # need to resize vertically -> no interpolation blur, exact bin.
    height = n_freq_bins

    arr = np.zeros((height, width), dtype=np.uint8)
    r0 = max(0, row_from_top - line_thickness // 2)
    r1 = min(height, r0 + line_thickness)
    arr[r0:r1, :] = 255
    Image.fromarray(arr, mode="L").save(path)

    return frequency


def synthesize_sine_wave(
    frequency_hz: float,
    duration_sec: float = 2.0,
    sr: int = 22050,
    fade_sec: float = 0.01,
) -> np.ndarray:
    """
    Generates a pure sine tone directly (no image involved) at the given
    frequency. Used for the "pick a note" option in the Audio -> Image
    direction, since there we want a real, clean reference tone to turn
    into a spectrogram image - not an image-based approximation of one.

    A short fade-in/fade-out is applied to avoid clicks at the start/end.
    """
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * frequency_hz * t)

    fade_samples = int(sr * fade_sec)
    if fade_samples > 0 and fade_samples * 2 < n_samples:
        fade_curve = np.linspace(0.0, 1.0, fade_samples)
        audio[:fade_samples] *= fade_curve
        audio[-fade_samples:] *= fade_curve[::-1]

    return audio.astype(np.float32)


if __name__ == "__main__":
    # Small end-to-end sanity check.
    # Generates test images, converts them to audio, and saves everything
    # in a subfolder "sanity_check_output".
    import os

    out_dir = "sanity_check_output"
    os.makedirs(out_dir, exist_ok=True)

    # Test 1: single tone
    line_img_path = os.path.join(out_dir, "test_single_line.png")
    make_test_image_single_line(line_img_path)
    audio, sr = image_to_audio(line_img_path, duration_sec=2.0)
    save_audio(audio, sr, os.path.join(out_dir, "test_single_line.wav"))
    print(f"Test 1 (single tone) written to {out_dir}/test_single_line.wav")

    # Test 2: harmonic series
    harmonic_img_path = os.path.join(out_dir, "test_harmonics.png")
    make_test_image_harmonic_stack(harmonic_img_path)
    audio, sr = image_to_audio(harmonic_img_path, duration_sec=2.0)
    save_audio(audio, sr, os.path.join(out_dir, "test_harmonics.wav"))
    print(f"Test 2 (harmonics) written to {out_dir}/test_harmonics.wav")

    # Test 3: pure tone tuned to a specific musical note
    note = "G#2"
    pure_tone_n_fft = 2048
    pure_tone_sr = 22050
    pure_tone_img_path = os.path.join(out_dir, "test_pure_tone.png")
    freq = make_test_image_pure_tone(
        pure_tone_img_path, note, sr=pure_tone_sr, n_fft=pure_tone_n_fft
    )
    audio, sr = image_to_audio(
        pure_tone_img_path,
        n_fft=pure_tone_n_fft,
        sr=pure_tone_sr,
        duration_sec=2.0,
    )
    save_audio(audio, sr, os.path.join(out_dir, "test_pure_tone.wav"))
    print(
        f"Test 3 (pure tone {note}, {freq:.2f} Hz) written to "
        f"{out_dir}/test_pure_tone.wav"
    )

    # Test the reverse direction: convert audio from Test 2 back into an image
    audio_to_image(
        os.path.join(out_dir, "test_harmonics.wav"),
        os.path.join(out_dir, "test_harmonics_reconstructed.png"),
    )
    print("Audio->Image reconstruction complete.")
