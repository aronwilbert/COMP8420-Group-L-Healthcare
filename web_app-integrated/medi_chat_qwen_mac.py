import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from pathlib import Path
from peft import PeftModel

import copy


# ======= Modules ========
from modules.ner import MedicalNERModule
from modules.retriever import RetrieverModule
from modules.redactor import CensorshipModule
from modules.specialty import stop_words, lemmatizer, preprocess_medical_transcript, remove_stopwords, lemmatize, TextPreprocessor, SpecialtyClassifierModule
from modules.sentiment import SentimentAnalysisModule

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title = 'MediAssist AI', page_icon = '🩺', layout = 'centered')

# ─────────────────────────────────────────────────────────────
# Sidebar – Settings
# ─────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

BASE_MODEL_NAME = 'Qwen/Qwen3-1.7B'
LORA_PATH = ROOT_DIR / 'LLM' / 'qwen3_1_7b_familydoctor_final'


SYSTEM_PROMPTS = {
    'BASELINE': 'You are a helpful medical assistant for a doctor. Keep your responses concise.',

    'TRANSCRIPTION_CHAT': (
        'You are a helpful medical assistant for a doctor. Your task is help the doctor analyzing transcriptions. Keep your responses concise. '
        'The user might provide you with some assisting question-answer pairs as context. You should check whether the information within these pairs is related to the user\'s query. '
        'The question-answer pairs are given within <context> and </context> tags. '
        'You should answer based on primarily your own knowledge and secondarily using the pairs that are relevant.'
    ),

    'PATIENT_CONNECT': (
        'You are a helpful medical assistant for a doctor. '
        'Your task is to help the doctor analyzing feedback messages from patients. '
        'The user(doctor) may provide you with the message from patient, analyze and provide response. '
        'Keep your responses concise.'
    ),

    'MEDICINE_EVALUATION': (
        'You are a helpful medical assistant for a doctor. Keep your responses concise. '
        'Your task is help the doctor analyzing medications. '
        'The user might provide you with some assisting question-answer pairs as context. You should check whether the information within these pairs is related to the user\'s query. '
        'The question-answer pairs are given within <context> and </context> tags. '
        'You should answer based on primarily your own knowledge and secondarily using the pairs that are relevant.'
    ),
}

REASONING      = True # Always ON
MAX_NEW_TOKENS = 2048 
TEMPERATURE = 0.10
TOP_P = 0.9

def reset_session():
    st.session_state.messages = []

    if 'ragcalled' in st.session_state:
        del st.session_state['ragcalled']

    if 'spec_detected' in st.session_state:
        del st.session_state['spec_detected']

    if 'sent_detected' in st.session_state:
        del st.session_state['sent_detected']

def set_app_state(new_state: dict):
    st.session_state['app_state'] = {
        'SYSTEM_PROMPT'      : new_state.get('SYSTEM_PROMPT',      SYSTEM_PROMPTS['BASELINE']),
        'PROMPT_PLACEHOLDER' : new_state.get('PROMPT_PLACEHOLDER', 'Ask me anything...'),
        'MODE_CAPTION'       : new_state.get('MODE_CAPTION',       'General'),
        'MEDICAL_NER_ON'     : new_state.get('MEDICAL_NER_ON', False),
        'MEDICAL_RAG_ON'     : new_state.get('MEDICAL_RAG_ON', False),
        'REDACTOR_ON'        : new_state.get('REDACTOR_ON',    False),
        'CLS_SPEC_ON'        : new_state.get('CLS_SPEC_ON',    False),
        'CLS_SENT_ON'        : new_state.get('CLS_SENT_ON',    False)
    }

if 'app_state' not in st.session_state:
    set_app_state(dict())

with st.sidebar:
    st.title('📚 Applications')

    if st.button('📝 Transcription Analysis', use_container_width = True):
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['TRANSCRIPTION_CHAT'],
            MEDICAL_NER_ON = True,
            MEDICAL_RAG_ON = True,
            REDACTOR_ON    = True,
            CLS_SPEC_ON    = True,
            PROMPT_PLACEHOLDER = 'Paste your transcription here...',
            MODE_CAPTION = '📝 Transcription Analysis',
        ))

    if st.button('💬 Patient Feedback Followup', use_container_width = True):
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['PATIENT_CONNECT'],
            MEDICAL_NER_ON = True,
            REDACTOR_ON    = True,
            CLS_SENT_ON    = True,
            PROMPT_PLACEHOLDER = 'Paste patient feedback message here...',
            MODE_CAPTION = '💬 Patient Feedback Followup',
        ))

    if st.button('💊 Medicine Board', use_container_width = True):
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['MEDICINE_EVALUATION'],
            MEDICAL_NER_ON = True,
            MEDICAL_RAG_ON = True,
            CLS_SPEC_ON    = True,
            PROMPT_PLACEHOLDER = 'Ask about any medicine...',
            MODE_CAPTION = '💊 Medicine Board',
        ))

    st.divider()

    st.info("""
    **Disclaimer**

    Educational use only.

    This chatbot is not a substitute for professional medical advice.
    """)

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
    """, unsafe_allow_html = True)

    st.divider()

    if st.button('Start Over', use_container_width = True):
        reset_session()
        set_app_state(dict())
        st.rerun()

SYSTEM_PROMPT      = st.session_state['app_state']['SYSTEM_PROMPT']
PROMPT_PLACEHOLDER = st.session_state['app_state']['PROMPT_PLACEHOLDER']
MODE_CAPTION       = st.session_state['app_state']['MODE_CAPTION']
MEDICAL_NER_ON     = st.session_state['app_state']['MEDICAL_NER_ON']
MEDICAL_RAG_ON     = st.session_state['app_state']['MEDICAL_RAG_ON']
REDACTOR_ON        = st.session_state['app_state']['REDACTOR_ON']
CLS_SPEC_ON        = st.session_state['app_state']['CLS_SPEC_ON']
CLS_SENT_ON        = st.session_state['app_state']['CLS_SENT_ON']

# ─────────────────────────────────────────────────────────────
# Model loading  (cached - survives Streamlit reruns)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='⏳ Loading fine-tuned Qwen3 FamilyDoctor model - this may take a few moments')
def load_environment(base_model_name: str, lora_path: str):
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f'LoRA model not found: {lora_path_obj}')

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup hardware device layout targets cleanly up-front
    if torch.cuda.is_available():
        device_mapping = {"": "cuda"}
        dtype          = torch.float16
    elif torch.backends.mps.is_available():
        device_mapping = {"": "mps"}
        dtype          = torch.float16
    else:
        device_mapping = {"": "cpu"}
        dtype          = torch.float32

    # Load base model directly into hardware layout memory (Removes .to(device) lag)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype               = dtype,          # Fixed deprecated argument name
        trust_remote_code   = True,
        device_map          = device_mapping, # Uses clean device pairing
        attn_implementation = "sdpa",         # Fast native Apple execution kernels
        local_files_only    = True            # Bypasses web-checks that hang loading
    )

    # Apply the fine-tuned medical weights directly over the base layout
    model = PeftModel.from_pretrained(base_model, str(lora_path_obj))
    model.eval()

    med_ner = MedicalNERModule()
    med_rag = RetrieverModule()
    redactor = CensorshipModule(placeholder = '███')
    cls_specialty = SpecialtyClassifierModule(str(APP_DIR / 'specialty_classifier_objects.joblib'))
    cls_sentiment = SentimentAnalysisModule()

    return model, tokenizer, med_ner, med_rag, redactor, cls_specialty, cls_sentiment


model, tokenizer, med_ner, med_rag, redactor, cls_specialty, cls_sentiment = load_environment(
    BASE_MODEL_NAME,
    str(LORA_PATH),
)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    reset_session()

# ─────────────────────────────────────────────────────────────
# Streaming generator
# ─────────────────────────────────────────────────────────────
_STRIP = ['<|im_end|>', '<|endoftext|>', '<|im_start|>'] 

RESP_STATE_INIT = 0
RESP_STATE_COT  = 1
RESP_STATE_RESP = 2

def model_device():
    return next(model.parameters()).device

def strip_highlighting(messages: list[dict[str, str]]):
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg['role'] == 'user':
            msg['content'] = med_ner.remove_highlighting(msg['content'])
    return messages

def strip_rag_docs_for_display(messages: list[dict[str, str]]):
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg['role'] == 'user':
            msg['content'] = med_rag.strip_rag_docs_for_display(msg['content'])
    return messages

def specialty_tag(label: str):
    return f':orange-badge[:material/star: Likely Medical Specialty - **{label}**]\n\n\n\n'

def sentiment_tag(label: str, score: str):
    score = f'{score*100:.2f}% possibility'
    if label == 'NEGATIVE':
        sentiment_text = f'Negative ({score})'
        color = 'red'
    elif label == 'POSITIVE':
        sentiment_text = f'Positive ({score})'
        color = 'green'
    else:
        sentiment_text = f'Neutral ({score})'
        color = 'yellow'

    return f':{color}-badge[:material/star: Tone of this message - **{sentiment_text}**]\n\n\n\n'

def stream_response(
    history:    list[dict],
    system:     str,
    think:      bool,
    max_tokens: int,
    temp:       float,
    tp:         float,
):
    messages = [
        { 'role': 'system', 'content': system }
    ] + history

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True, enable_thinking = think)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

    inputs = tokenizer(prompt, return_tensors = 'pt').to(model_device())

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt                  = True,
        skip_special_tokens          = False, 
        clean_up_tokenization_spaces = False,
    )

    gen_kwargs = {
        **inputs,
        'streamer':       streamer,
        'max_new_tokens': max_tokens,
        'temperature':    max(temp, 1e-6),
        'do_sample':      temp > 0.0,
        'top_p':          tp,
        'pad_token_id':   tokenizer.eos_token_id,
    }

    thread = Thread(target = model.generate, kwargs = gen_kwargs, daemon = True) 
    thread.start()

    buffer   = ''
    thinking = ''
    response = ''
    t_start  = 0 
    t_end    = 0 
    state: int = RESP_STATE_INIT

    for chunk in streamer:
        buffer += chunk

        if state == RESP_STATE_INIT:
            if think:
                state = RESP_STATE_COT
                if '<think>' in chunk:
                    t_start = buffer.index('<think>') + len('<think>')
                else:
                    t_start = 0
            elif buffer.strip() and not buffer.lstrip().startswith('<'):
                state = RESP_STATE_RESP

        if state == RESP_STATE_COT:
            if '<think>' in chunk:
                t_start = buffer.index('<think>') + len('<think>')
            if '</think>' in chunk:
                t_end    = buffer.index('</think>')
                thinking = buffer[t_start:t_end]
                response = buffer[t_end + len('</think>'):].lstrip('\n')
                state    = RESP_STATE_RESP
            else:
                thinking = buffer[t_start:]

        if state == RESP_STATE_RESP:
            if t_end:
                response = buffer[t_end + len('</think>'):].lstrip('\n')
            else:
                response = buffer

            for tok in _STRIP:
                response = response.replace(tok, '')

        yield thinking, response, False

    for tok in _STRIP:
        response = response.replace(tok, '')
    yield thinking.strip(), response.strip(), True

    thread.join()

# ─────────────────────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────────────────────
st.title('🩺 MediAssist AI')
st.caption(MODE_CAPTION)

def change_censorship_character(text: str):
    if not REDACTOR_ON:
        return text
    return text.replace(redactor.placeholder, '███')

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):

        if msg['role'] == 'assistant' and msg.get('thinking'):
            with st.expander('💭 Thought Process', expanded = False):
                st.markdown(msg['thinking'])

        if MEDICAL_RAG_ON and msg['role'] == 'user':
            msg['content'] = med_rag.strip_rag_docs_for_display(msg['content'])

        if msg['role'] == 'user':
            user_content = msg['content']
            if 'rendered_ner' in msg:
                user_content = msg['rendered_ner']

            if 'specialty' in msg:
                user_content = specialty_tag(msg['specialty']) + user_content

            if 'sentiment' in msg:
                user_content = sentiment_tag(msg['sentiment'][0], msg['sentiment'][1]) + user_content

            st.markdown(user_content, unsafe_allow_html = True)
        else:
            st.markdown(msg['content'], unsafe_allow_html = True)

if user_input := st.chat_input(PROMPT_PLACEHOLDER):

    with st.spinner('Working on your request...'):
        spec_added = False
        if CLS_SPEC_ON:
            if 'spec_detected' not in st.session_state:
                st.session_state['spec_detected'] = cls_specialty(user_input)['prediction']
                spec_added = True

        sent_added = False
        if CLS_SENT_ON:
            if 'sent_detected' not in st.session_state:
                st.session_state['sent_detected'] = cls_sentiment(user_input)
                sent_added = True

        if REDACTOR_ON:
            user_input = redactor(user_input)

        user_input_for_display = user_input

        new_user_message = {
            'role':   'user',
            'content': user_input
        }

        if MEDICAL_RAG_ON:
            if 'ragcalled' not in st.session_state:
                documents = med_rag(new_user_message['content'])
                new_user_message['content'] = med_rag.apply_rag_template(new_user_message['content'], documents)
                st.session_state['ragcalled'] = True

    st.session_state.messages.append(new_user_message)

    with st.chat_message('user'):
        if MEDICAL_NER_ON:
            entities, modifiers, doc = med_ner.extract_entities_and_modifiers(user_input_for_display)
            user_input_for_display = med_ner.apply_highlighting(user_input_for_display, entities, modifiers)
            new_user_message['rendered_ner'] = user_input_for_display

        if spec_added:
            user_input_for_display = specialty_tag(st.session_state['spec_detected']) + user_input_for_display
            new_user_message['specialty'] = st.session_state['spec_detected']

        if sent_added:
            snt_label, snt_score = st.session_state['sent_detected']
            user_input_for_display = sentiment_tag(snt_label, snt_score) + user_input_for_display
            new_user_message['sentiment'] = st.session_state['sent_detected']

        st.markdown(user_input_for_display, unsafe_allow_html = True)

    with st.chat_message('assistant'):
        think_exp  = st.expander('💭 Thinking...', expanded = True)
        think_slot = think_exp.empty()   
        resp_slot  = st.empty()          

        last_thinking = ''
        last_response = ''

        message_history = st.session_state.messages

        for thinking, response, is_final in stream_response(
            history    = message_history,
            system     = SYSTEM_PROMPT,
            think      = REASONING,
            max_tokens = MAX_NEW_TOKENS,
            temp       = TEMPERATURE,
            tp         = TOP_P,
        ):
            last_thinking = thinking
            last_response = response

            if thinking:
                think_slot.markdown(thinking)

            resp_slot.markdown(response if is_final else response + ' ▌')

        think_slot.markdown(last_thinking)
        resp_slot.markdown(last_response)

    st.session_state.messages.append({
        'role':     'assistant',
        'content':  last_response,
        'thinking': last_thinking,
    })