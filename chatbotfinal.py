import streamlit as st
import pandas as pd
import base64
from PyPDF2 import PdfReader
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
pdf_path='treatments-autism_508.pdf'
# ---------- PDF Handling ----------     
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# ---------- Vector Store ----------
def get_vector_store(text_chunks):
    # Using pretrained HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store, embeddings

# ---------- Conversational Chain ----------
def get_conversational_chain():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "Answer is not available in the context."

Context: {context}
Question: {question}
Answer:
"""
    )
    model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return chain

# ---------- User Input ----------
def user_input(user_question, pdf_path, conversation_history):
    # if not pdf_path:
    #     st.warning("Please upload a PDF first.")
    #     return

    # Process PDF and get vector store
    text_chunks = get_text_chunks(get_pdf_text(pdf_path))
    vector_store, embeddings = get_vector_store(text_chunks)

    # Load vector store
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    # Get answer from chain
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response['output_text']

    # Update conversation history
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conversation_history.append((user_question, answer, "llama-3.3-70b-versatile", timestamp, pdf_path))

    # Display chat
    for q, a, model_name, ts, pdf_name in reversed(conversation_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}\n---")

    # Save conversation history as CSV
    df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.sidebar.markdown(f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history</button></a>', unsafe_allow_html=True)

# ---------- M
