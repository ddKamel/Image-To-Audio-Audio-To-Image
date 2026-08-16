## Sound-to-Image | Image to Sound

This project is a showcase of a classic approach from Sound-To-Image
and Image-To-Sound for the course of Computational Modelling.

## Installation and Running

For installing the project locally simply run <br>

```pip install -r requirements.txt```

To run it on your local browser run it with Streamlit

```streamlit run app.py```

A demo can be found here:

https://image-to-audio-audio-to-image-bjonftzuytan7x2pnglvaf.streamlit.app/

## Functionality
Image-to-Audio: We can upload any image and "hear" what it kind of sounds like. If we just treat the pixel values as amplitude we can code them into sounds.
There we can tweak different settings and see how they actually change the sound. (see tooltips)
There are two options for using already prefabricated images: One with a single tone and one with harmonics.

Audio-to-Image: The other way around. We can upload any audio and see what it looks like.
There is also an option to choose a certain piano note and its octave and generate an image out of that.
Additionally, there is an option to apply a lowpass, highpass filter and distortion to add some variation to these notes.

Paint-Spectrogram: We can also paint with a simple brush tool and an eraser.
There is also an option to add a harmonic stack with overtones already added.


## How does it work?

The general logic differs between functions.
So let's begin with the Image-To-Audio procedure.

### Image-To-Audio

1. Open the image as a greyscale. Since we are using mono audio only colour doesn't carry any meaningful information.
2. Compute the target size of the image. We can tweak this value by determining the desired duration of the audio and by choosing a different window size for FFT (n_fft)
<br> For more information about the size of the frequency bands read: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
3. Scale the brightness values to dB (lowest value=silence, highest = 0 dB)
4. Griffin-Lim algorithm to reconstruct the phase. This uses iSTFT and STFT.
<br> For more information about the algorithm: https://learnius.com/slp/9+Speech+Synthesis/1+Fundamental+Concepts/7+Waveform+Generation/Griffin-Lim+algorithm
5. Peak normalization: Griffin-Lim often ends up with a low volume so we normalize it back

<b> Why Griffin-Lim </b> <br>
Griffin-Lim is a relatively fast algorithm which reconstructs our phase by mathematical estimation.
Since our converted images only consist of the amplitude spectrum we are missing lots of information in the time-domain.
Without the phase spectrum speech can sound metallic or robotic and miss some important peaks.
Of course this algorithm has its limitations since it lacks of any statistical priors of f.e. human speech.
But for the purpose of constructing and reconstructing in our cases it is good enough.

Even though Griffin-Lim is not the state of the art anymore and has been widely replaced with VoCoders first and then Diffusion Neural Networks
it has found its comeback in some modern Speech Synthesis Neural networks as initial phase estimation.

### Audio-To-Image

The Audio-To Image Pipeline is a little more straightforward:

1. Load the audio, convert it to mono (with set sampling rate)
2. Perform STFT onto the loaded audio: The result is our frequency band x timeframe matrix
3. Calculate the magnitude using np.abs from the previous result
Read more about the maths: https://numpy.org/devdocs/reference/generated/numpy.absolute.html
4. Scale our amplitude to dB. We also clip very silent signals according to our set dynamic dB range so we don't hear certain artifacts.
5. Normalize, flip and save as image!

Here we do not use any algorithms like Griffin-Lim. Keep in mind that during this process the phase information is lost.
The sampling rate is standard set in the code at 22050. Most music is produced at around 44,1kHz. 
So why does it still sound relatively accurate?

The Nyquist-Shannon-Theorem tells us that the highest frequency we can reconstruct is half the sample rate.
So we can easily reconstruct everything until 11kHz. This is enough for human speech and music.
Of course some details like high frequency percs will be lost, but the effect is relatively mild.
By setting a higher frequency rate and a higher n_fft window size we can also map even higher frequencies (up to 22kHz for standard music)
The trade of is that the width of the image would be bigger and thus use more memory. For the purpose of this project our set sample rate is sufficient.
But you are free to experiment with what happens when the chosen values are not optimally set.

To be as accurate as possible we could always save some original information like the sampling rate inside the metadata of the .png header or construct some extra pixels which can be read out later (or ignored if missing).


## Libaries used

To make everything as smooth and as easily runable as possible we use the libraries librosa (for stft and algorithms) as well as streamlit for the webhosting.
Pillow and soundfile give us some useful helper functions.
The drawable canvas is made using the streamlit-image-coordinates and scipy library.

## Mentions

The coding was assisted by Claude Code.
