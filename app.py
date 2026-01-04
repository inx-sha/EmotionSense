# app.py
import streamlit as st
import joblib
import numpy as np
import librosa
from sentence_transformers import SentenceTransformer

st.title("EmotionSense — Speech & Text Emotion Recognition (MVP)")

# load models
@st.cache_resource
def load_models():
    audio_obj = joblib.load('models/audio_svm.joblib')
    text_obj = joblib.load('models/text_clf.joblib')
    embedder = SentenceTransformer(text_obj['embedder'])
    return audio_obj['model'], text_obj['clf'], embedder

audio_model, text_clf, embedder = load_models()

st.header("Text input")
txt = st.text_area("Enter text")
if st.button("Predict text emotion"):
    if not txt.strip():
        st.warning("Enter some text")
    else:
        emb = embedder.encode([txt])
        pred = text_clf.predict(emb)[0]
        probs = text_clf.predict_proba(emb)[0]
        st.success(f"Predicted emotion: {pred}")
        st.write("Class probabilities:", dict(zip(text_clf.classes_, map(float, probs))))

st.header("Speech input")
uploaded = st.file_uploader("Upload WAV audio (mono, <=3s recommended)", type=['wav','mp3'])
if uploaded is not None:
    # save temp
    import tempfile, soundfile as sf
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tf.write(uploaded.getvalue())
    tf.flush()
    # extract features
    mf = librosa.feature.mfcc(y=librosa.load(tf.name, sr=22050, duration=3)[0], sr=22050, n_mfcc=40)
    feat = np.concatenate([np.mean(mf,axis=1), np.std(mf,axis=1)]).reshape(1,-1)
    pred = audio_model.predict(feat)[0]
    probs = audio_model.predict_proba(feat)[0]
    st.success(f"Predicted emotion: {pred}")
    st.write("Class probabilities:", dict(zip(audio_model.classes_, map(float, probs))))
