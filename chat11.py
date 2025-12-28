# ================================
# OPTIMIZED CHATBOT (FAST VERSION)
# ================================

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import base64
from dotenv import load_dotenv

@st.cache_data
def get_base64_image(image_file:str)->str:
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 🌄 Background Image
img_base64 = get_base64_image("brain.jpg")

st.markdown(f"""
    <style>
    [class="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
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

# ---------------------------------------------------------
# CACHE PDF READING (prevents loading on every message)
# ---------------------------------------------------------
@st.cache_resource
def load_pdf():
    docs = "treatments-autism_508.pdf"
    reader = PdfReader(docs)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

# ---------------------------------------------------------
# CACHE EMBEDDINGS + VECTOR DB (BIGGEST SPEED BOOST)
# ---------------------------------------------------------
@st.cache_resource
def load_vector_store(pages):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_texts(pages, embedding=embeddings)

# ---------------------------------------------------------
# CACHE LLM (Groq initialization is expensive)
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")


# ========================
# SIDEBAR CHATBOT SECTION
# ========================
with st.sidebar:

    st.title("🤖 NeuroBot")

    load_dotenv()
    os.environ["GROQ_API_KEY"] = "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"

    # Load everything ONCE instead of every message
    pages = load_pdf()
    vector_store = load_vector_store(pages)
    llm = load_llm()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant answering questions concisely.")
        ]

    # Show existing chat messages
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # User input
    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve related docs
        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in docs])

        # Prepare final prompt
        template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the context to answer concisely (max 7 sentences).\n\n"
                "Context: {context}\n\n"
                "Question: {question}"
            ),
        )

        final_prompt = template.format(context=context, question=prompt)

        # Get response from LLM (fast now)
        result = llm.invoke(final_prompt)
        answer = result.content if hasattr(result, "content") else str(result)

        st.session_state.messages.append(AIMessage(content=answer))

        with st.chat_message("assistant"):
            st.markdown(answer)
