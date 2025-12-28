import streamlit as st
import base64
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert your local image to base64
img_base64 = get_base64_image("image.jpg")
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
        color: #00bfff;  /* 🔵 Sky blue color */
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        margin-top: 200px;
        font-family: 'Poppins', sans-serif;
    }}
    </style>
    <h1>⚕️ Autism Detection & Chatbot System</h1>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .merge-text {
        text-align: center;
        font-size: 2.4rem;
        line-height: 1.8;
        color: #e0e0de;
        margin-top: 25px;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-shadow: 3px 3px 12px rgba(0, 0, 0, 0.8);
        animation: fadeIn 2s ease-in-out;
    }

    /* Title (optional enhancement) */
    h1 {
        color: #f5f5f5;
        text-align: center;
        font-size: 3.2rem;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.6);
        margin-top: 180px;
        font-weight: 100;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <p class="merge-text">
        <span class="gradient-text">Autism is not a disease.</span><br>
        <span class="gradient-text">It's a different way of being human.</span>
    </p>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .merge-text {
        text-align: center;
        font-size: 2.4rem;
        line-height: 1.8;
        color: #e0e0de;
        margin-top: 25px;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-shadow: 3px 3px 12px rgba(0, 0, 0, 0.8);
        animation: fadeIn 2s ease-in-out;
    }

    /* Title (optional enhancement) */
    h1 {
        color: #f5f5f5;
        text-align: center;
        font-size: 3.2rem;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.6);
        margin-top: 180px;
        font-weight: 1000;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <p class="merge-text">
        <span class="gradient-text">Autism Spectrum Disorder (ASD) is not an illness — it’s a neurological difference 
        that affects how people communicate, learn, and interact with the world around them. 
        Every individual with autism has unique strengths and challenges. </span><br>
        <span class="gradient-text">With awareness, understanding, and support, we can help create an inclusive environment 
        where everyone can thrive.</span>
    </p>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .glow-btn {
        display: inline-block;
        padding: 14px 35px;
        font-size: 1.2rem;
        font-family: 'Poppins', sans-serif;
        color: white;
        background: linear-gradient(90deg, #ff758c, #ff7eb3);
        border: none;
        border-radius: 35px;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255, 120, 150, 0.6);
        margin-top: 20px;
    }

    .glow-btn:hover {
        transform: scale(1.07);
        box-shadow: 0 0 25px rgba(255, 180, 200, 0.8);
        background: linear-gradient(90deg, #ff9a9e, #fad0c4);
    }
    </style>
""", unsafe_allow_html=True)


