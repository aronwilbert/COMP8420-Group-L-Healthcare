import streamlit as st 
from datetime import datetime 

st.set_page_config(
    page_title = "Healthcare LLM Chatbot", 
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
    max-width: 1100px;
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
    """
        Replace with function laeter
    """
    return (
        "Place holder"
        f"Your input was : **{user_message}**"
    )

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
    st.header("🩺,HealthCare LLM")

    st.markdown("""
    <div class="info-card">
    <b>Model</b><br>
    BioGPT <br><br>
    <b>Mode</b><br>
    Conversational assistant<br><br>
    <b>Status</b><br>
    Prototype UI
    </div<
""", unsafe_allow_html = True)
    
    st.divider() 
    
    st.info("""
    **Disclaimer** 
            
    Educational use only. 
    
    This chatbot is not a substitute for professional medical advice.
    """)

    st.divider() 
    st.subheader("Sample Prompts")

    samples = {
        "Drug REview": "I have been taking panadol for headache "
    }

    for label, sample_text in samples.items():
        if st.button(label, use_container_width = True):
            st.session_state.selected_sample = sample_text
    
    st.divider()

    if st.button("Clear Chat", use_container_width = True):
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Chat cleared. How can i help today?", 
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

#Header 

st.markdown("""
<h1 style = 'text-align:center'>🩺 Healthcare LLM Chatbot</h1>
<p style = 'text-align:center; color:gray;'>
Privacy-aware medical text assistant powered by BioGPT
<p>
""",unsafe_allow_html = True)

st.write("")

# Chat area 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        st.caption(message.get("time", ""))

# Input Handling 
prompt = st.chat_input("Ask Healthcare LLm!")

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
