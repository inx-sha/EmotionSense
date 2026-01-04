# src/audio_model.py 
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def extract_mfcc(path, n_mfcc=40, sr=22050, duration=3, offset=0.0):
    y, _ = librosa.load(path, sr=sr, duration=duration, offset=offset)
    # pad or trim
    if len(y) < sr*duration:
        pad_len = sr*duration - len(y)
        y = np.pad(y, (0, pad_len))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # aggregate: mean and std across time axis
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def load_audio_dataset(base_dir):
    X, y = [], []
    for label in os.listdir(base_dir):
        lab_dir = os.path.join(base_dir, label)
        if not os.path.isdir(lab_dir): continue
        for fname in os.listdir(lab_dir):
            if not fname.endswith('.wav'): continue
            path = os.path.join(lab_dir, fname)
            X.append(extract_mfcc(path))
            y.append(label)
    return np.array(X), np.array(y)

def train_and_save(base_dir, model_out='models/audio_svm.joblib'):
    X, y = load_audio_dataset(base_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump({'model': clf}, model_out)
    print('Saved', model_out)

if __name__ == '__main__':
    # adjust data path
    train_and_save('data/audio')
