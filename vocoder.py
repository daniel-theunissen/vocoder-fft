import pyaudio
import numpy as np
import time
import scipy.signal as signal

p = pyaudio.PyAudio()

CHANNELS = 1
RATE = 44100
b, a = signal.ellip(4, 0.01, 120, 0.125)

def callback(in_data, frame_count, time_info, flag):
    # using Numpy to convert to array for processing
    audio_data = np.fromstring(in_data, dtype=np.float32)
    audio_data /= 2
    return audio_data, pyaudio.paContinue

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(20)
    stream.stop_stream()
    print("Stream is stopped")

stream.close()

p.terminate()