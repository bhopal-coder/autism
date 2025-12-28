import streamlit as st
import base64
# import test
import os
import subprocess
import joblib
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import base64
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import streamlit.components.v1 as components
if "current_page" not in st.session_state:
    st.session_state.current_page="home"
def main():
    st.session_state.current_page="home"
def page1():
    st.session_state.current_page="test"
@st.cache_data
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# 🌄 Background Image
if st.session_state.current_page=="home":

    img_base64 = get_base64_image("back.jpg")

    # 🌟 Page Styling
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;     
            background-repeat: no-repeat;
            height: 100vh;
        }}
        h1 {{
            color: #00bfff;
            text-align: center;
            font-size: 3rem;
            font-weight: 900;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
            margin-top: 200px;
            font-family: 'Poppins', sans-serif;
        }}
        .merge-text {{
            text-align: center;
            font-size: 2.4rem;
            line-height: 1.8;
            color: #e0e0de;
            margin-top: 25px;
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            text-shadow: 3px 3px 12px rgba(0, 0, 0, 0.8);
            animation: fadeIn 2s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .button-wrapper {{
            display: flex;
            justify-content: center;
            margin-top: 50px;
        }}
        div.stButton > button:first-child {{
            padding: 14px 40px !important;
            font-size: 1.3rem !important;
            font-family: 'Poppins', sans-serif !important;
            color: white !important;
            background: linear-gradient(90deg, #ff758c, #ff7eb3) !important;
            border: none !important;
            border-radius: 35px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 20px rgba(255, 120, 150, 0.6) !important;
        }}
        div.stButton > button:first-child:hover {{
            transform: scale(1.07) !important;
            box-shadow: 0 0 25px rgba(255, 180, 200, 0.8) !important;
            background: linear-gradient(90deg, #ff9a9e, #fad0c4) !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # 🌸 Title and Text
    st.markdown(
        """
        <h1 style='color: pink;'>⚕️ Autism Detection & Chatbot System</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <p class="merge-text">
        Autism Spectrum Disorder (ASD) is not an illness — it’s a neurological difference 
        that affects how people communicate, learn, and interact with the world around them.<br><br>
        Every individual with autism has unique strengths and challenges.<br>
        With awareness, understanding, and support, we can help create an inclusive environment 
        where everyone can thrive.
    </p>
    """, unsafe_allow_html=True)

    # 🌟 Glowing Streamlit Button (centered under text)
    st.markdown('<div class="button-wrapper">', unsafe_allow_html=True)
    if st.button("Proceed to Chatbot 💬", key="chatbot_glow"):
        st.switch_page("pages/test.py")
        st.rerun()
# if st.session_state.current_page=="test":
    