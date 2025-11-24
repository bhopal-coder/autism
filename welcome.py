import streamlit as st
import pandas as pd
import joblib
import numpy as np
import subprocess
import threading
import time
import socket

from sklearn.preprocessing import LabelEncoder
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="autism")

# Add a fixed bottom-right button using HTML
 
df=pd.read_csv('Autismdata.csv')
st.title("Autism Detection")
chatbot_file = "chat.py"  # your second Streamlit file
chatbot_port = "8502"        # port for chatbot
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
# --- Function to run chatbot.py on another port ---
def run_chatbot():
    subprocess.Popen(
        ["streamlit", "run", chatbot_file, "--server.port", chatbot_port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# --- Start chatbot in background if not already started ---
if "chatbot_started" not in st.session_state:
    st.session_state.chatbot_started = False

if not st.session_state.chatbot_started:
    threading.Thread(target=run_chatbot, daemon=True).start()
    st.session_state.chatbot_started = True
    time.sleep(2)

# --- Floating Chat Button + Frame ---
# st.markdown(f"""
# <style>
# .chat-button {{
#     position: fixed;
#     bottom: 20px;
#     right: 20px;
#     background-color: #4CAF50;
#     color: white;
#     border: none;
#     border-radius: 50%;
#     width: 60px;
#     height: 60px;
#     font-size: 26px;
#     cursor: pointer;
#     box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
#     z-index: 1000;
#     transition: 0.3s;
# }}
# .chat-button:hover {{
#     background-color: #45a049;
#     transform: scale(1.1);
# }}
# .chat-frame {{
#     display: none;
#     position: fixed;
#     bottom: 90px;
#     right: 20px;
#     width: 50vw;
#     height: 80vh;
#     border-radius: 12px;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.3);
#     z-index: 999;
#     overflow: hidden;
#     background: white;
# }}
# .close-chat {{
#     position: absolute;
#     top: 8px;
#     right: 10px;
#     border: none;
#     background: #ff4d4d;
#     color: white;
#     border-radius: 50%;
#     width: 28px;
#     height: 28px;
#     font-size: 16px;
#     cursor: pointer;
#     z-index: 1001;
# }}
# </style>

# <button class="chat-button" onclick="openChatbot()">💬</button>

# <div id="chatbot-frame" class="chat-frame">
#     <button class="close-chat" onclick="document.getElementById('chatbot-frame').style.display='none';">✖</button>
#     <iframe id="chatbot-iframe" src="" width="100%" height="100%" style="border:none; border-radius:12px;"></iframe>
# </div>

# <script>
# function openChatbot() {{
#     var frame = document.getElementById('chatbot-frame');
#     var iframe = document.getElementById('chatbot-iframe');
#     frame.style.display = 'block';
#     iframe.src = 'http://localhost:{chatbot_port}';
# }}
# </script>
# """, unsafe_allow_html=True)
st.markdown("""
<a href="http://localhost:8502" target="_blank">
    <button style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 26px;
        cursor: pointer;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    ">💬</button>
</a>
""", unsafe_allow_html=True)

    
st.header("Fill your details:")
name=st.text_input("Enter name:")          
age=st.text_input("Enter age:")            
                                           
options=['Agree','Disagree']
st.header("Autism Spectrum Disorder Test:")
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
#   st.write(output)
  if output==1:
    st.write("Autism present")
    st.write("For any treatments or suggestions you can ask from the bot")

  else:
    st.write("Autism absent")
# accuracy_score(y_test,y_pred)

  









