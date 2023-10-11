import csv
import os
import random

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
import librosa


# func to extract MFCC features
def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs


# func for speed and pitch augmentation
def speed_pitch_aug(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    audio_rate = random.uniform(0.7, 1.3)
    pitch_shift = random.uniform(-3, 3)
    audio_speed = librosa.effects.time_stretch(audio, rate=audio_rate)
    audio_pitch = librosa.effects.pitch_shift(audio_speed, sr=sample_rate, n_steps=pitch_shift)
    return audio_pitch


# func to add noise
def add_noise(audio, min_level=0.001, max_level=0.05):
    level = random.uniform(min_level, max_level)
    noise = np.random.normal(0, level, len(audio))
    added_noise = audio + noise
    return added_noise


# func to min-max scaling
def min_max(audio):
    min_val, max_val = np.min(audio), np.max(audio)
    scaled_audio = (audio - min_val) / (max_val - min_val)
    return scaled_audio


# func for padding/trimming
def pad_trim(audio, length):
    if len(audio) > length:
        return audio[:length]
    else:
        return np.pad(audio, (0, length - len(audio)))


# dir of recorded samples
data_dir = "../data_collection/voice_dataset/"

# dir to save extracted features
feature_dir = "preprocessed_data"
os.makedirs(feature_dir, exist_ok=True)

individual_name = {"Tavish": 1, "Other": 0}

num_augmentation = 50

print("creating labels.csv")

# create csv files to store labels
csv_filename = "labels.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["individual", "feature_path"])

    for individual in individual_name.keys():
        for original_sample in os.listdir(f'{data_dir}{individual}'):
            if original_sample.endswith(".wav"):
                original_audio_path = os.path.join(f'{data_dir}{individual}', original_sample)

                mfcc_feature_original = extract_mfcc(original_audio_path)

                # print(f"MFCC feature shape: {mfcc_feature_original.shape}")

                mfcc_original_filename = f"{original_sample}_mfcc.npy"
                mfcc_path_original = os.path.join(feature_dir, mfcc_original_filename)
                np.save(mfcc_path_original, mfcc_feature_original)

                csv_writer.writerow([individual_name[individual], mfcc_path_original])

                for i in range(num_augmentation):
                    augmented_audio = original_audio_path

                    # apply speed/pitch variation and noise
                    augmented_audio = speed_pitch_aug(augmented_audio)
                    augmented_audio = add_noise(augmented_audio)

                    # apply norm
                    norm_audio = min_max(augmented_audio)

                    # apply pad/trim
                    padded_audio = pad_trim(norm_audio, 220500)

                    temp_wav_path = f"temp_augmented_{i}.wav"
                    sf.write(temp_wav_path, padded_audio, 44100)
                    mfcc_feature = extract_mfcc(temp_wav_path)

                    # print(f"MFCC feature shape: {mfcc_feature.shape}")
                    os.remove(temp_wav_path)
                    mfcc_filename = f"{original_sample}_aug_{i}_mfcc.npy"
                    mfcc_path = os.path.join(feature_dir, mfcc_filename)
                    np.save(mfcc_path, mfcc_feature)

                    csv_writer.writerow([individual_name[individual], mfcc_path])

print("data preprocessing completed...")

# load labels from csv
df = pd.read_csv(csv_filename)

# split data into training, validation, and testing sets
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

train_df.to_csv("train_labels.csv", index=False)
val_df.to_csv("val_labels.csv", index=False)
test_df.to_csv("test_labels.csv", index=False)

print("data splits saved and completed...")
