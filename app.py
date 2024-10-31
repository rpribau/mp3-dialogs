import os
import streamlit as st
import whisper
import torch
from datetime import datetime
import pandas as pd
import re
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar la variable de entorno XDG_RUNTIME_DIR
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-dir'
if not os.path.exists(os.environ['XDG_RUNTIME_DIR']):
    os.makedirs(os.environ['XDG_RUNTIME_DIR'])

# Verifica si tienes una GPU disponible para Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"

# Función para extraer las características MFCC
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Función para predecir emociones usando el modelo de emociones
def predict_emotion(audio_path, model):
    mfcc_features = extract_mfcc(audio_path)
    X = np.array([mfcc_features])
    X = np.expand_dims(X, -1)
    
    prediction = model.predict(X)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    predicted_emotion = emotions[np.argmax(prediction)]
    probability = np.max(prediction)
    
    return predicted_emotion, probability

def process_files(files, emotion_model):
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    data = []

    for file in files:
        try:
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Extraer la fecha del nombre del archivo si está presente
            # Reemplaza esta sección de código en process_files:
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file.name)
            if date_match:
                try:
                    creation_date = datetime.strptime(date_match.group(0), "%Y%m%d")
                except ValueError:
                    creation_date = datetime.now()  # Si el formato de fecha es incorrecto
            else:
                creation_date = datetime.now()  # Si no hay fecha en el nombre del archivo, usa la fecha actual


            if file.name.endswith('.mp3'):
                audio_path = temp_file_path
            elif file.name.endswith('.mp4'):
                video = VideoFileClip(temp_file_path)
                audio_path = os.path.join(temp_dir, f"{os.path.splitext(file.name)[0]}.wav")
                video.audio.write_audiofile(audio_path)
                video.close()
            else:
                continue

            # Cargar el modelo Whisper en el dispositivo correspondiente y transcribir
            whisper_model = whisper.load_model("small", device=device)
            result = whisper_model.transcribe(audio_path)
            transcript = result['text']

            # Predecir la emoción
            emotion, probability = predict_emotion(audio_path, emotion_model)

            data.append({
                'Date': creation_date,
                'Transcript': transcript,
                'Filename': file.name,
                'Emotion': emotion,
                'Probability (%)': round(probability * 100, 2)
            })

            os.remove(temp_file_path)
            if file.name.endswith('.mp4'):
                os.remove(audio_path)

        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")

    df = pd.DataFrame(data)
    return df

# Configuración de la app en Streamlit
st.markdown("<h1 style='text-align: center; font-size: 48px;'>Extraer diálogos y emociones de audios y videos</h1>", unsafe_allow_html=True)

st.markdown("""
## Instrucciones:
1. **Sube tus archivos MP3 o MP4**: Usa el botón para seleccionar y subir múltiples archivos MP3 o MP4.
2. **Procesar Archivos**: Haz clic en el botón "Procesar Archivos" para extraer las transcripciones y emociones.
3. **Revisar y Descargar**: Revisa los resultados y descarga el archivo CSV.

### Herramientas Utilizadas:
- **Streamlit**: Para la interfaz.
- **OpenAI Whisper**: Para transcripciones.
- **Modelo de Emociones**: Para predecir la emoción en el audio.
- **MoviePy y Librosa**: Para manejar archivos multimedia.
- **Pandas**: Para manejo de datos.
""")

uploaded_files = st.file_uploader("Sube archivos .mp3 o .mp4", accept_multiple_files=True)

if st.button("Procesar Archivos"):
    if uploaded_files:
        # Cargar el modelo de emociones antes de procesar archivos
        emotion_model = load_model('modelo_lstm.h5')
        df = process_files(uploaded_files, emotion_model)
        st.dataframe(df)

        # Proveer una opción para descargar el CSV
        csv = df.to_csv(index=False)
        st.download_button(label="Descargar CSV", data=csv, file_name="transcripts_emotions.csv", mime="text/csv")
    else:
        st.warning("Por favor, sube algunos archivos primero.")
