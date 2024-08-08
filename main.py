import streamlit as st
import whisper
import tempfile
import os

# Cargar modelo de Whisper
model = whisper.load_model("base")

st.title("Aplicación de Transcripción de Audio/Video")

# Cargador de archivos
uploaded_file = st.file_uploader("Elegí un archivo de audio/video...", type=["mp3", "mp4", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.audio(uploaded_file, format="audio/mp3")

    # Transcripción
    st.write("Transcribiendo...")
    result = model.transcribe(temp_file_path, verbose=True, language="es", task="transcribe")

    st.write("¡Transcripción completa!")
    st.text_area("Transcripción", result["text"], height=300)

    # Proveer enlace de descarga para la transcripción
    st.download_button("Descargar Transcripción", result["text"], file_name="transcripción.txt")

    # Eliminar archivo temporal
    os.remove(temp_file_path)
