# main.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Page config (must be at top) ---
st.set_page_config(page_title="Autism Portal", layout="wide")

# --- Header ---
st.title("🌈 Autism Awareness Portal")
st.image("chatbot.jpg", width=60)

# --- Load data ---
# Make sure Autismdata.csv is in the same folder or provide full path
df = pd.read_csv("Autismdata.csv")

# --- Chat toggle in session_state ---
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# Floating chat button that toggles session state
if st.button("💬 Open Chatbot" if not st.session_state.chat_open else "❌ Close Chatbot"):
    st.session_state.chat_open = not st.session_state.chat_open

# If open, show iframe (only once, not twice)
if st.session_state.chat_open:
    # Use a div wrapper with fixed positioning so CSS obeys bottom/right values
    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 120px;   /* adjust vertical position */
            right: 40px;     /* adjust horizontal position */
            width: 380px;
            height: 500px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            overflow: hidden;
        ">
            <iframe src="http://localhost:8502" width="100%" height="100%" style="border:none; border-radius:12px;"></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.header("Fill your details:")

# --- Form inputs ---
name = st.text_input("Enter name:")
age_input = st.text_input("Enter age:")

st.header("Autism Spectrum Disorder Test:")

options = ["Agree", "Disagree"]
# Create radios with keys to persist answers
q1 = st.radio("Q1: I often notice small sounds when others do not", options, key="q1")
q2 = st.radio("Q2: When I read a story, I find it difficult to work out characters' intentions.", options, key="q2")
q3 = st.radio("Q3: I find it easy to read between the lines when someone is talking to me.", options, key="q3")
q4 = st.radio("Q4: I usually concentrate more on the whole picture, rather than the small details.", options, key="q4")
q5 = st.radio("Q5: I know how to tell if someone listening to me is getting bored.", options, key="q5")
q6 = st.radio("Q6: I find it easy to do more than one thing at once.", options, key="q6")
q7 = st.radio("Q7: I find it easy to work out what someone is thinking or feeling just by looking at their face.", options, key="q7")
q8 = st.radio("Q8: If there is an interruption, I can switch back to what I was doing very quickly.", options, key="q8")
q9 = st.radio("Q9: I like to collect information about categories of things.", options, key="q9")
q10 = st.radio("Q10: I find it difficult to work out people's intentions.", options, key="q10")

sex_option = st.radio("Select Gender:", ["Male", "Female"], key="sex")
q11 = st.radio("Q11: Have you suffered from jaundice till now?", options, key="q11")
q12 = st.radio("Q12: Has your family ever suffered from this syndrome?", options, key="q12")

# --- Validate inputs ---
missing = []
if not age_input:
    missing.append("age")

# ensure all radio questions are answered (they will be, but keep check)
for i, q in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12], start=1):
    if q not in options:
        missing.append(f"q{i}")

if missing:
    st.warning("Please fill all required fields before predicting.")
else:
    # convert age to int safely
    try:
        age = int(age_input)
        if age < 0 or age > 120:
            st.warning("Please enter a valid age (0-120).")
            age = None
    except ValueError:
        st.warning("Age must be a number.")
        age = None

# If age invalid, stop here
if age is None:
    st.stop()

# --- Prepare feature encoding ---
# Use explicit mapping for Agree/Disagree so it's stable
binary_map = {"Agree": 1, "Disagree": 0}
q_vals = [
    binary_map[q1],
    binary_map[q2],
    binary_map[q3],
    binary_map[q4],
    binary_map[q5],
    binary_map[q6],
    binary_map[q7],
    binary_map[q8],
    binary_map[q9],
    binary_map[q10],
]

# For Sex, Jaundice and Family_ASD, fit encoders based on df values (keeps consistency)
# Make sure df has columns 'Sex', 'Jauundice' (typo in dataset) and 'Family_ASD' as you used earlier
# Adjust column names if they are spelled differently in your CSV
sex_col = "Sex"
jaundice_col = "Jauundice"  # if actual column name is 'Jaundice' change this
family_col = "Family_ASD"

# Fit encoders based on dataset unique values
enc_sex = LabelEncoder().fit(df[sex_col].astype(str))
enc_jau = LabelEncoder().fit(df[jaundice_col].astype(str))
enc_fam = LabelEncoder().fit(df[family_col].astype(str))

sex_val = enc_sex.transform([sex_option])[0]
jau_val = enc_jau.transform([q11])[0]
fam_val = enc_fam.transform([q12])[0]

# Build final input vector in same order as training features
# Original order in your earlier code was: q1..q10, age, sex, jaundice, family
input_vector = q_vals + [age, sex_val, jau_val, fam_val]
valuef = np.array(input_vector).reshape(1, -1).astype(int)

# --- Load or train model (load if pkl exists) ---
model_path = "autism_detection.pkl"
try:
    model = joblib.load(model_path)
    # optional: inform user
    st.info("Loaded pretrained model.")
except FileNotFoundError:
    st.info("No saved model found — training a new one from dataset. This may take a moment.")
    # Prepare training data from df (assuming the same columns & ordering)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=50, criterion="gini", random_state=42)
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)
    st.success("Model trained and saved to autism_detection.pkl")

# --- Prediction button ---
if st.button("Predict"):
    try:
        output = model.predict(valuef)
        # adjust output interpretation depending on how target is encoded in your dataset
        if int(output[0]) == 1:
            st.write("🔴 Autism present")
            st.write("For treatments or suggestions you can ask the bot (💬).")
        else:
            st.write("🟢 Autism absent")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
