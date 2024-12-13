import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time

# Record the modulator audio
print("Recording modulator audio...")
duration = 5  # Record duration in seconds
sr = 44100  # Sampling rate
modulator_audio = np.zeros((int(duration * sr),))  # Initialize empty array

# Clear buffer and wait for a bit
time.sleep(1)  # Ensure proper initialization
modulator_audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()  # Block until recording is complete
modulator_audio = modulator_audio.flatten()  # Convert to 1D array
sf.write('modulator_recorded.wav', modulator_audio, sr)  # Save the recorded modulator audio
print("Modulator audio recorded and saved as 'modulator_recorded.wav'.")

# Load the carrier and modulator (recorded) signals
carrier, sr = librosa.load('carrier.wav', sr=None)
modulator, sr = librosa.load('modulator_recorded.wav', sr=None)

# Ensure both signals are the same length
min_len = min(len(carrier), len(modulator))
carrier = carrier[:min_len]
modulator = modulator[:min_len]

def frame_signal(signal, frame_size, hop_size):
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.lib.stride_tricks.as_strided(
        signal, shape=(num_frames, frame_size),
        strides=(signal.strides[0] * hop_size, signal.strides[0])
    )
    return frames

frame_size = 1024
hop_size = 512

carrier_frames = frame_signal(carrier, frame_size, hop_size)
modulator_frames = frame_signal(modulator, frame_size, hop_size)

def stft(frames, n_fft):
    return np.fft.rfft(frames, n=n_fft)

n_fft = 1024

carrier_stft = stft(carrier_frames, n_fft)
modulator_stft = stft(modulator_frames, n_fft)

modulator_amplitude = np.abs(modulator_stft)
modulated_stft = carrier_stft * (modulator_amplitude / (np.abs(carrier_stft) + 1e-10))

def istft(stft_matrix, hop_size):
    num_frames, n_fft = stft_matrix.shape
    frame_size = (n_fft - 1) * 2
    signal = np.zeros(num_frames * hop_size + frame_size - hop_size)
    for n, i in enumerate(range(0, len(signal) - frame_size, hop_size)):
        signal[i:i + frame_size] += np.fft.irfft(stft_matrix[n])
    return signal

output_signal = istft(modulated_stft, hop_size)
output_signal = output_signal / np.max(np.abs(output_signal))
sf.write('stereo_file1.wav', output_signal, 44100, 'PCM_24')

# Plot DFT of the modulator audio
modulator_dft = np.fft.fft(modulator)
frequencies = np.fft.fftfreq(len(modulator), 1/sr)

plt.figure(figsize=(10, 6))
plt.semilogx(frequencies[:len(frequencies)//2], np.abs(modulator_dft)[:len(modulator_dft)//2])
plt.title("DFT of Modulator Audio")
plt.xlabel("Frequency (Hz, Log Scale)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
