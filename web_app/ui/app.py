import streamlit as st 
import sys
from pathlib import Path 
from datetime import datetime 

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from backend.llm_service import generate_answer


st.set_page_config(
    page_title = "MediAssist AI", 
    page_icon = "🩺",
    layout = "centered"
)


# Styling 

st.markdown(""" 
<style>
.main{
    background-color: #f7f9fb;
}
.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem; 
    max-width: 760px;
}
.chat-title{
    font-size: 2.3rem; 
    font-weight: 800; 
    color: #12355b;             
}  
.subtitle{
    color: #5c677d; 
    padding: 1rem; 
    border-radius: 14px; 
    border: 1px solid #e5e7eb; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.disclaimer{
    background: #fff7ed; 
    color: #9a3412; 
    padding: 0.8rem; 
    border-radius: 12px;
    border: 1px solid #fed7aa; 
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html = True)

#LLM connector 

def call_llm(user_message: str) -> str:
    return generate_answer(user_message)

# Session state 

if "messages" not in st.session_state: 
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content":(
                "Hello! I am your Healthcare Assistant. "
                "Ask a Healthcare - related question!"
            ),
            "time": datetime.now().strftime("%H:%M")
        }
    ]

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = ""

# SideBar 

with st.sidebar:
    st.header("🩺 MediAssist AI")

    st.markdown("""
    <div class="info-card">
    <b>Model</b><br>
    BioGPT <br><br>
    <b>Mode</b><br>
    Conversational assistant<br><br>
    </div>
""", unsafe_allow_html = True)
    
    st.divider() 
    
    st.info("""
    **Disclaimer** 
            
    Educational use only. 
    
    This chatbot is not a substitute for professional medical advice.
    """)

    st.subheader("Quick Actions")

    if st.button("💊 Medication Side Effects", use_container_width=True):
        st.session_state.selected_sample = "What are common side effects of Panadol?"
        st.rerun()

    if st.button("🤒 Fever and Cough", use_container_width=True):
        st.session_state.selected_sample = "I have fever, cough, and fatigue for three days. What should I do?"
        st.rerun()

    if st.button("🩺 Symptom Guidance", use_container_width=True):
        st.session_state.selected_sample = "I have a headache and mild dizziness. What could this mean?"
        st.rerun()

    if st.button("❤️ Heart Attack Warning Signs", use_container_width=True):
        st.session_state.selected_sample = "What are the early symptoms of a heart attack?"
        st.rerun()

    st.divider()

    if st.button("Clear Chat", use_container_width = True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hi, How can i help today?", 
                "time": datetime.now().strftime("%H:%M")
            }
        ]
        st.rerun()
    
    st.divider()

    st.markdown("""
        <div style="
        text-align:center;
        color:gray;
        font-size:12px;
        padding-top:20px;
        ">
        🩺 Healthcare NLP System<br>
        COMP8420 Advanced NLP<br>
        Educational Use Only
        </div>
    """, unsafe_allow_html=True)
    st.success("BioGPT backend connected")

#Header 

st.markdown("""
<h1 style = 'text-align:center'>🩺 MediAssist AI</h1>
<p style = 'text-align:center; color:gray;'>
Privacy-aware medical text assistant powered by BioGPT
<p>
""",unsafe_allow_html = True)

st.write("")

st.markdown("""
<div style="
background:#1f2937;
padding:25px;
border-radius:16px;
border:1px solid #374151;
margin-bottom:20px;">

<h3>🩺 Welcome to MediAssist AI</h3>
<p>
Get reliable health information, medication guidance, and symptom-related insights powered by BioGPT.
</p>
<p>
✓ Symptom Guidance &nbsp;&nbsp;
✓ Medication Information &nbsp;&nbsp;
✓ Health Education &nbsp;&nbsp;
✓ Privacy-Aware Responses
</p>

</div>
""", unsafe_allow_html=True)
# Chat area 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        st.caption(message.get("time", ""))

# Input Handling 
prompt = st.chat_input("Ask MediAssist AI!")

if st.session_state.selected_sample:
    prompt = st.session_state.selected_sample 
    st.session_state.selected_sample =  ""

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content":prompt, 
        "time": datetime.now().strftime("%H:%M")
    })

    with st.chat_message("user"):
        st.write(prompt)
        st.caption(datetime.now().strftime("%H:%M"))
    
    with st.chat_message("assistant"):
        with st.spinner("LLM is generating response..."):
            response = call_llm(prompt)
            st.write(response)
            st.caption(datetime.now().strftime("%H:%M"))
    
    st.session_state.messages.append({
        "role":"assistant", 
        "content": response, 
        "time": datetime.now().strftime("%H:%M")
    })
