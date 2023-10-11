import torch
import pyaudio
import soundfile as sf
import noisereduce as nr
import numpy as np
import librosa
from model.model import VoiceAuthenticationBinaryClassifier

# Load the trained model
model = VoiceAuthenticationBinaryClassifier(num_mfcc=13, num_frames=431)
model.load_state_dict(torch.load('training/voice_authentication_binary_model.pth'))
model.eval()

# Set a threshold for authentication
threshold = 0.5
audio = pyaudio.PyAudio()


def record_audio(filename, time=5):
    input("press enter to say password...")
    print("listening...")
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


def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    data, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs


def main():
    # Record the user's voice
    record_audio("sample.wav")

    # Reduce noise
    audio_data, sample_rate = sf.read("sample.wav")
    rn_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

    # Save noise-reduced audio
    sf.write("sample.wav", rn_audio, sample_rate)

    # Preprocess the recorded voice
    preprocessed_data = extract_mfcc("sample.wav")

    # Convert the preprocessed data to a PyTorch tensor
    input_data = torch.from_numpy(preprocessed_data).unsqueeze(0)  # Add batch dimension
    input_data = input_data.unsqueeze(1)

    mean = input_data.mean()
    std = input_data.std()
    input_data = (input_data - mean) / std

    # Make a prediction using the model
    with torch.no_grad():
        prediction = model(input_data)

    print(prediction.item())
    # Authenticate the user
    # if prediction.item() > threshold:
    #     print(f'authentication successful: {round(prediction.item(), 2)*100}% voice match')
    # else:
    #     print(f'authentication failed: {round(prediction.item(), 2)*100}% voice match')


if __name__ == "__main__":
    main()
