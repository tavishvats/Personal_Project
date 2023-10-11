import os
import soundfile as sf
import noisereduce as nr
import numpy as np
import pyaudio
import csv

# create directory
dataset_dir = "voice_dataset"
os.makedirs(dataset_dir, exist_ok=True)

num_samples = 20
audio = pyaudio.PyAudio()


def record_audio(filename, time=5):
    print(f"recording audio to {filename}...")
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    for x in range(0, int(44100 / 1024 * time)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    w_file = sf.SoundFile(filename, 'w', samplerate=44100, channels=1, format='wav')

    data = np.frombuffer(b''.join(frames), dtype=np.int16)
    w_file.write(data)
    w_file.close()

# I change the individual name to "Other" when recording unauthorized samples.
individual_name = "Saurav"

individual_dir = os.path.join(dataset_dir, individual_name)
os.makedirs(individual_dir, exist_ok=True)

csv_filename = os.path.join(dataset_dir, "metadata.csv")
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    print(f"recording voice sample for {individual_name}:")

    for i in range(num_samples):
        input("press enter to start recording...")
        file = os.path.join(individual_dir, f"{individual_name}_{i + 1}.wav")
        record_audio(file)
        print(f"saved: {file}")

        # handling noise
        audio_data, sample_rate = sf.read(file)
        print(sample_rate)
        rn_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

        # saving noise reduced audio
        sf.write(file, rn_audio, sample_rate)

        # write metadata
        csv_writer.writerow([individual_name, file])

print(f"Voice samples for {individual_name} recorded.")
audio.terminate()
