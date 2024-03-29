import streamlit as st
from PIL import Image
import warnings
from gtts import gTTS
from io import BytesIO
import os
import google.generativeai as genai

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = "sua chave"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

# Importações após a supressão de avisos
from moondream import Moondream, detect_device
from transformers import CodeGenTokenizerFast as Tokenizer

def model_inference(image) -> str:
    # Definindo o prompt padrão
    prompt = "You are an assistant and you can describe the environment around a visually impaired person, identifying objects, people, and obstacles, and even their characteristics, such as color and relative position."
    device, dtype = detect_device()
    model_id = "vikhyatk/moondream1"
    tokenizer = Tokenizer.from_pretrained(model_id)
    moondream = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
    img = Image.open(image)
    image_embeds = moondream.encode_image(img)
    answer = moondream.answer_question(image_embeds, prompt, tokenizer)
    return answer


def translater(description):
    model = genai.GenerativeModel(model_name="gemini-pro")
    safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                ]
    response = model.generate_content("Traduza para o português este texto"+description,
                    safety_settings=safety_settings,
                )
    answer = response.text

    return answer


# Função para converter texto em áudio e exibir no Streamlit
def text_to_speech(text):
    tts = gTTS(text=text, lang='pt')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Interface Streamlit
st.title('Assistente de Visão e Descrição de Imagens')

# Nota para o usuário sobre a captura de imagem
st.write("Por favor, use uma imagem previamente capturada com a câmera do seu dispositivo ou carregue uma imagem.")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer:
    if st.button('Descrever a imagem'):
        description = model_inference(img_file_buffer)        
        text_translater = translater(description)
        st.write("Descrição:", text_translater)        
        audio_file = text_to_speech(text_translater)
        st.audio(audio_file, format='audio/mp3')

# Carregar imagem
uploaded_image = st.file_uploader("Escolha uma imagem", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    if st.button('Descrever Imagem'):
        if uploaded_image is not None:
            # Mostrar imagem
            st.image(uploaded_image, caption='Imagem Carregada', use_column_width=True)
            
            # Executar inferência
            description = model_inference(uploaded_image)
            text_translater = translater(description)
            st.write("Descrição:", text_translater)        
            audio_file = text_to_speech(text_translater)
            st.audio(audio_file, format='audio/mp3')              
        else:
            st.write("Por favor, carregue uma imagem.")
