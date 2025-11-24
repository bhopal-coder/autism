import streamlit as st
import pandas as pd
import joblib
import numpy as np
import subprocess
import threading
import time
import socket
import chat1111
import os
import streamlit as st
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
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False
st.set_page_config(page_title="autism")



  # more balanced width


# Add a fixed bottom-right button using HTML
 
df=pd.read_csv('Autismdata.csv')

# st.title("Autism Detection")?
st.markdown("<h1 style='color:white;'>Autism Detection</h1>", unsafe_allow_html=True)

    

    # chatbot_file = "chat1111.py"  # your second Streamlit file
    # chatbot_port = "8502"        # port for chatbot
    # def is_port_in_use(port):
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         return s.connect_ex(("localhost", port)) == 0
    # # --- Function to run chatbot.py on another port ---
    # def run_chatbot():
    #     subprocess.Popen(
    #         ["streamlit", "run", chatbot_file, "--server.port", chatbot_port],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.DEVNULL
    #     )



        
# st.header("Fill your details:")
st.markdown("<h2 style='color:white;'>Fill your details:</h2>", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Change the label color */
    label, .stTextInput label {
        color: white !important;
    }

    /* Change the text color inside the input box */
    .stTextInput input {
        color: white !important;
        background-color: #262730; /* optional dark background */
    }
    </style>
""", unsafe_allow_html=True)
name=st.text_input("Enter name:")          
age=st.text_input("Enter age:")   
options=['Agree','Disagree']         
st.markdown("<h2 style='color:white;'>Autism Disorder Test</h2>", unsafe_allow_html=True)                                      

st.markdown("""
    <style>
        /* Change radio question text */
        .stRadio > label {
            color: white !important;
            font-weight: 600 !important;
        }

        /* Change radio options text */
        .stRadio div[role='radiogroup'] label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

if "q1" not in st.session_state:
    st.session_state.q1 = None

q1 = st.radio(
    "Question 1: I often notice small sounds when others do not",
    options,
    index=0 if st.session_state.q1 else None,
    key="q1_radio"
)
if "q2" not in st.session_state:
    st.session_state.q2 = None

q2 = st.radio(
    "Question 2: When I am reading a story, I find it difficult to work out the characters' intentions.",
    options,
    index=0 if st.session_state.q2 else None,
    key="q2_radio"
)
# q1 = st.radio(
#     "Question 1: I often notice small sounds when others do not",
#     ("Agree", "Disagree")  # Options
# )

if "q3" not in st.session_state:
    st.session_state.q3 = None
q3 = st.radio(
    "Question 3: I find it easy to read between the lines when someone is talking to me.",
options,
    index=0 if st.session_state.q3 else None,
    key="q3_radio" # Options
)
if "q4" not in st.session_state:
    st.session_state.q4 = None
q4 = st.radio(
    "Question 4:I usually concentrate more on the whole picture, rather than the small details.",
    options,
    index=0 if st.session_state.q4 else None,
    key="q4_radio"
)

if "q5" not in st.session_state:
    st.session_state.q5 = None
q5 = st.radio(
    "Question 5:I know how to tell if someone listening to me is getting bored.",
    options,
    index=0 if st.session_state.q5 else None,
    key="q5_radio" # Options
)

if "q6" not in st.session_state:
    st.session_state.q6 = None
q6 = st.radio(
    "Question 6: I find it easy to do more than one thing at once.",
    options,
    index=0 if st.session_state.q6 else None,
    key="q6_radio" # Options
)

if "q7" not in st.session_state:
    st.session_state.q7 = None
q7 = st.radio(
    "Question 7: I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    options,
    index=0 if st.session_state.q7 else None,
    key="q7_radio" # Options
)
if "q8" not in st.session_state:
    st.session_state.q8 = None
q8 = st.radio(
    "Question 8: If there is an interruption, I can switch back to what I was doing very quickly.",
    options,
    index=0 if st.session_state.q8 else None,
    key="q8_radio"
)
if "q9" not in st.session_state:
    st.session_state.q9= None
q9 = st.radio(
    "Question 9: I like to collect information about categories of things.",
    options,
    index=0 if st.session_state.q9 else None,
    key="q9_radio"# Options
)
if "q10" not in st.session_state:
    st.session_state.q10 = None
q10 = st.radio(
    "Question 10: I find it difficult to work out people's intentions.",
    options,
    index=0 if st.session_state.q10 else None,
    key="q10_radio"  # Options
)

option=['Male','Female']
if "q" not in st.session_state:
    st.session_state.q = None

q = st.radio(
    "Select Gender:",
    option,
    index=0 if st.session_state.q else None,
    key="q_radio"
)
if "q11" not in st.session_state:
    st.session_state.q11 = None
q11 = st.radio(
    "Question 11: Have you suffered from jaundice till now?",
    options,
    index=0 if st.session_state.q11 else None,
    key="q11_radio"  # Options
)
if "q12" not in st.session_state:
    st.session_state.q12 = None
q12 = st.radio(
    "Question 12: Has your family ever suffered from this syndrome?",
    options,
    index=0 if st.session_state.q12 else None,
    key="q12_radio"  # Options
)

encoder_ad=LabelEncoder()
encoder_gen=LabelEncoder()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
df['Sex']=encoder_ad.fit_transform(df['Sex'])
df['Jauundice']=encoder_ad.fit_transform(df['Jauundice'])
df['Family_ASD']=encoder_ad.fit_transform(df['Family_ASD'])
df['Class']=encoder_ad.fit_transform(df['Class'])
encoder_ad.fit(['Disagree', 'Agree'])

encoder_gen.fit(['Male','Female'])
if not q1 or not q2 or not q3 or not q4 or not q5 or not q6 or not q7 or not q8 or not q9 or not q10 or not q11 or not q12:
    print("⚠️ Please fill all inputs before proceeding.")
# q is scalar 
else:
    q1=encoder_ad.transform([q1])[0]
    q1=1-q1
    q2=encoder_ad.transform([q2])[0]
    q2=1-q2
    q3=encoder_ad.transform([q3])[0]
    q3=1-q3
    q4=encoder_ad.transform([q4])[0]
    q4=1-q4
    q5=encoder_ad.transform([q5])[0]
    q5=1-q5
    q6=encoder_ad.transform([q6])[0]
    q6=1-q6
    q7=encoder_ad.transform([q7])[0]
    q7=1-q7
    q8=encoder_ad.transform([q8])[0]
    q8=1-q8
    q9=encoder_ad.transform([q9])[0]
    q1=1-q9
    q10=encoder_ad.transform([q10])[0]
    q10=1-q10
    q=encoder_gen.transform([q])[0]
    q=1-q
    q11=encoder_ad.transform([q11])[0]
    q11=1-q11
    q12=encoder_ad.transform([q12])[0]
    q12=1-q12
x=df.iloc[:,:-1]
value=(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,age,q,q11,q12)
valuef=np.array(value).reshape(1,-1)
import pandas as pd
valuef = pd.DataFrame(valuef).fillna(0).to_numpy()
#independent x
y=df.iloc[:,-1]                           
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#model                                    
model=RandomForestClassifier(n_estimators=50,criterion='gini')
model.fit(x_train,y_train)

if (st.button("Predict")):
    model=joblib.load('autism_detection.pkl')
    output=model.predict(valuef)
    from chat1111 import run_chatbot_ui

    if output == 1:
        st.success("✅ Autism detected.")
        st.info("For any treatments or suggestions, you can ask NeuroBot in the sidebar.")
    else:
        st.success("🧠 Autism not detected.")

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

    
    #   st.write(output)
    #   if output==1:
    #     st.write("Autism present")
    #     st.write("For any treatments or suggestions you can ask from the bot")
    #     with st.sidebar:
    #             st.header("🧠 NeuroBot Assistant")
    #             st.markdown(
    #                 f"""
    #                 <iframe src="http://localhost:{chatbot_port}" 
    #                         width="100%" 
    #                         height="600" 
    #                         style="border:none; border-radius:10px;">
    #                 </iframe>
    #                 """, 
    #                 unsafe_allow_html=True
    #             )

        

    #   else:
    #     st.write("Autism absent")
    # accuracy_score(y_test,y_pred)
    
    









