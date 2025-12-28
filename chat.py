import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# st.set_page_config(page_title="Groq Chatbot")
st.title("🤖 Chatbot")

# -----------------------------
# Example documents
# -----------------------------
os.environ["GROQ_API_KEY"] = "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"

# docs = [
#     "Autism is a developmental disorder affecting communication and behavior.",
#     "Early diagnosis of autism improves outcomes through therapy.",
#     "Therapies for autism include behavioral therapy, speech therapy, and occupational therapy."
# ]
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
vector_store = Chroma.from_texts(docs, embedding=embeddings)

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

    # Extract string safely
    # if isinstance(raw_response, dict):
    #     answer_text = raw_response.get("text") or raw_response.get("content") or str(raw_response)
    # else:
    #     answer_text = str(raw_response)
    if hasattr(raw_response, "content"):
        answer_text = raw_response.content
    elif isinstance(raw_response, dict):   
        answer_text = raw_response.get("content") or raw_response.get("text") or str(raw_response)
    else:
        answer_text = str(raw_response)

    # -----------------------------zz
    # Add to chat
    # -----------------------------
    st.session_state.messages.append(AIMessage(content=answer_text))
    with st.chat_message("assistant"):
        st.markdown(answer_text)
