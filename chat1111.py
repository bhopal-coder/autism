import os
import streamlit as st
# from dotenv import load_dotenv

# LangChain imports

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
import base64
@st.cache_data
def get_base64_image(image_file:str)->str:
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 🌄 Background Image
img_base64 = get_base64_image("brain.jpg")

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
# Load environment variables
def run_chatbot_ui():

    # load_dotenv()
    # st.set_page_config(page_title="Groq Chatbot")
    st.title("🤖NeuroBot")

    # -----------------------------
    # Example documents
    # -----------------------------
    os.environ["GROQ_API_KEY"] = "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"

    docs="treatments-autism_508.pdf"
    reader = PdfReader(docs)
    docss = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            docss.append(text)



    # -----------------------------
    # Initialize embeddings + Chroma vector store
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(docss, embedding=embeddings)

    # -----------------------------
    # Initialize chat history
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant answering questions concisely.")
        ]

    # Display previous messages
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # -----------------------------
    # User input
    # -----------------------------
    prompt = st.chat_input("Ask your question...")

    if prompt:
        # Add user message to chat
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # -----------------------------
        # Initialize Groq LLM with a supported model
        # -----------------------------
        llm = ChatGroq(model="llama-3.3-70b-versatile")  # replace with a model you have access to

        # -----------------------------
        # Retrieve relevant documents
        # -----------------------------
        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        relevant_docs = retriever.invoke(prompt)  # use get_relevant_documents
        context = "\n\n".join([d.page_content for d in relevant_docs])

        # -----------------------------
        # Create the prompt manually
        # -----------------------------
        template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                # "You are a helpful assistant.\n"
                "Use the context below to answer the question concisely (max 7 sentences).\n\n"
                "Context: {context}\n\n"
                "Question: {question}"
            ),
        )

        full_prompt = template.format(context=context, question=prompt)

        # -----------------------------
        # Call the LLM
        # -----------------------------
        raw_response = llm.invoke(full_prompt)
        if hasattr(raw_response, "content"):
            answer_text = raw_response.content
        elif isinstance(raw_response, dict):
            answer_text = raw_response.get("content") or raw_response.get("text") or str(raw_response)
        else:
            answer_text = str(raw_response)
        
        st.session_state.messages.append(AIMessage(content=answer_text))
        with st.chat_message("assistant"):
            st.markdown(answer_text)
         
         