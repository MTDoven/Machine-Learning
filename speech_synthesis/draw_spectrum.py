import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = r"C:\Users\t1526\Desktop\xxx.wav"
y, sr = librosa.load(file_path)

D = np.abs(librosa.stft(y))
DB = librosa.amplitude_to_db(D, ref=np.max)

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 8))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Tacotron 2")
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.show()
