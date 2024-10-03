import subprocess
import streamlit as st
import whisper
from datetime import datetime
import pandas as pd
import os
import re
from moviepy.editor import VideoFileClip

def process_files(files):
    # Asegúrate de que el directorio temporal exista
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Inicializar una lista vacía para almacenar los datos
    data = []

    for file in files:
        try:
            temp_file_path = os.path.join(temp_dir, file.name)
            
            # Guardar el archivo temporalmente en el disco
            with open(temp_file_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Extraer la fecha del nombre del archivo si está presente (por ejemplo, "audio_20230902.mp3")
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file.name)
            if date_match:
                creation_date = datetime.strptime(date_match.group(0), "%Y%m%d")
            else:
                creation_date = datetime.now()  # Si no hay fecha en el nombre del archivo, usa la fecha actual

            # Comprobar si el archivo es MP3 o MP4
            if file.name.endswith('.mp3'):
                audio_path = temp_file_path
            elif file.name.endswith('.mp4'):
                # Extraer audio de MP4
                video = VideoFileClip(temp_file_path)
                audio_path = os.path.join(temp_dir, f"{os.path.splitext(file.name)[0]}.mp3")
                video.audio.write_audiofile(audio_path)
                video.close()
            else:
                continue  # Si el archivo no es MP3 o MP4, saltarlo

            # Cargar el modelo Whisper y transcribir el archivo
            model = whisper.load_model("small")
            result = model.transcribe(audio_path)
            transcript = result['text']

            # Añadir la fecha, la transcripción y el nombre del archivo a la lista
            data.append({'Date': creation_date, 'Transcript': transcript, 'Filename': file.name})

            # Opcionalmente, eliminar los archivos temporales después de procesarlos
            os.remove(temp_file_path)
            if file.name.endswith('.mp4'):
                os.remove(audio_path)
                
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")

    # Crear un DataFrame a partir de la lista de datos
    df = pd.DataFrame(data)
    return df

# Componentes de la GUI de Streamlit
st.markdown("<h1 style='text-align: center; font-size: 48px;'>Extraer diálogos de audios MP3 y videos MP4</h1>", unsafe_allow_html=True)

st.markdown("""
## Instrucciones:
1. **Sube tus archivos MP3 o MP4**: Usa el botón de abajo para seleccionar y subir múltiples archivos MP3 o MP4.
2. **Procesar Archivos**: Haz clic en el botón "Procesar Archivos" para extraer las transcripciones.
3. **Revisar y Descargar**: Revisa las transcripciones resultantes y descarga el archivo CSV.

### Herramientas Utilizadas:
- **Streamlit**: Para la creación de la interfaz de usuario.
- **OpenAI Whisper**: Para la transcripción de los archivos de audio.
- **MoviePy**: Para extraer el audio de archivos MP4.
- **Pandas**: Para manejar y manipular los datos tabulares.
- **Python**: Como lenguaje de programación principal.

---

### Créditos:
**Desarrollado por**: Roberto Priego Bautista  
**GitHub**: [@rpribau](https://github.com/rpribau)
""")

uploaded_files = st.file_uploader("Sube archivos .mp3 o .mp4", accept_multiple_files=True)

if st.button("Procesar Archivos"):
    if uploaded_files:
        df = process_files(uploaded_files)
        st.dataframe(df)

        # Proveer una opción para descargar el CSV
        csv = df.to_csv(index=False)
        st.download_button(label="Descargar CSV", data=csv, file_name="transcripts.csv", mime="text/csv")
    else:
        st.warning("Por favor, sube algunos archivos primero.")