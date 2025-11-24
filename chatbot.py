import streamlit as st
import pandas as pd
import base64
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
#to get text chunks from pdf:
pdf_docs='treatments-autism_508.pdf'
# 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

def get_pdf_text(pdf_docs):
    text=""
    # for pdf in pdf_docs:
    pdf_reader=PdfReader(pdf_docs)
    for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
    #to get chunks from text
def get_text_chunks(text, model_name):
    if model_name=="llama-3.3-70b-versatile":
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=600)
    chunks=text_splitter.split_text(text)
    return chunks
# AIzaSyDQOzS3bziYZ1i2NARrg4n1RcsEnujykE0
# google_api_key='AIzaSyDQOzS3bziYZ1i2NARrg4n1RcsEnujykE0'
def get_vector_store(text_chunks,model_name,api_key=None):
    if model_name=='llama-3.3-70b-versatile':
      embeddings=HuggingFaceEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
# print("API Key:", google_api_key)



# create a conversational chain using langchain
def get_conversational_chain(model_name,vectorstore=None,api_key=None):
     if model_name=="llama-3.3-70b-versatile":
            prompt_template = PromptTemplate(
             input_variables=["context"],
             template="""
              
              Answer the question as detailed as possible from the provided context, make sure to provide all the
              details with proper structure,if the answer is not provided context just say,"answer is not
              available in
               
             the context", don't provide the wrong answer\n\n
              Context:\n {context}\n
              Question:\n{question}\n
              Answer:
              """
            )
            model=ChatGroq(model='llama-3.3-70b-versatile',temperature=0.3,google_api_key='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')
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
    vector_store = get_vector_store(text_chunks,model_name,api_key)
    user_question_output = ""              
    response_output = ""                  
    # updated to huggingface embedding    
    if model_name == "llama-3.3-70b-versatile":
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("llama-3.3-70b-versatile", vectorstore=new_db, api_key='AIzaSyDQOzS3bziYZ1i2NARrg4n1RcsEnujykE0')
        # huggingface update
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )
    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1 :   
        last_item = conversation_history[-1] 
        conversation_history.remove(last_item)
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])

        # df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64. convert to downloadable link 
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        #href mein link aa jayega downloadble files ka
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.snow()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    # linkedin_profile_link = "https://www.linkedin.com/in/snsupratim/"
    # kaggle_profile_link = "https://www.kaggle.com/snsupratim/"
    # github_profile_link = "https://github.com/snsupratim/"

    model_name = "llama-3.3-70b-versatile"
    api_key ="AIzaSyDQOzS3bziYZ1i2NARrg4n1RcsEnujykE0"
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            
            # api_key = None  # Reset Google API key
            # pdf_docs = None  # Reset PDF 
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")
        pdf_docs = 'treatments-autism_508.pdf'
        # if st.button("Submit & Process"):
        #     # if pdf_docs:
            #     with st.spinner("Processing..."):
            #         st.success("Done")  
            # else:
            #     st.warning("Please upload PDF files before processing.")

        user_question = st.text_input("Ask a Question")

        if user_question:
            user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
            st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()

    
