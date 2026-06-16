import streamlit as st
import os
import warnings
import transformers.utils.import_utils as transformers_import_utils

# Keep the ML/NLP stack single-process in Streamlit. On Python 3.13, joblib/loky
# can otherwise leave a harmless semaphore warning at shutdown.
os.environ.setdefault('JOBLIB_MULTIPROCESSING', '0')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

warnings.filterwarnings(
    'ignore',
    message=r'resource_tracker: There appear to be .* leaked semaphore objects.*',
    category=UserWarning,
)

# This app is text-only. In the Python 3.13 Miniconda environment, an
# incompatible torchvision install can break PEFT via transformers imports.
transformers_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from time import time
import copy
from pathlib import Path
from peft import PeftModel

# ======= Modules ========
from modules.ner import MedicalNERModule
from modules.retriever import RetrieverModule
from modules.redactor import CensorshipModule
from modules.specialty import stop_words, lemmatizer, preprocess_medical_transcript, remove_stopwords, lemmatize, TextPreprocessor, SpecialtyClassifierModule
# from modules.sentiment import SentimentAnalysisModule
from modules.sentiment_analyser import stop_words, lemmatizer, normalize_text, remove_stopwords, lemmatize, TextPreprocessor, SentimentAnalysisModule

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
    'BASELINE': 'You are MediAssist AI, a helpful medical assistant. Keep your responses concise.',

    'TRANSCRIPTION_CHAT': (
        'You are MediAssist AI, a helpful medical assistant for a doctor. Your task is help the doctor analyzing transcriptions. Keep your responses concise. '
        'The user might provide you with some assisting question-answer pairs as context. You should check whether the information within these pairs is related to the user\'s query. '
        'The question-answer pairs are given within <context> and </context> tags. '
        'You should answer based on primarily your own knowledge and secondarily using the pairs that are relevant.'
    ),

    'PATIENT_CONNECT': (
        'You are MediAssist AI, a helpful medical assistant for a doctor. '
        'Your task is to help the doctor analyzing feedback messages from patients. '
        'The user(doctor) may provide you with the message from patient, analyze and provide response. '
        'Keep your responses concise.'
    ),

    'MEDICINE_EVALUATION': (
        'You are MediAssist AI, a helpful medical assistant for a doctor. Keep your responses concise. '
        'Your task is help the doctor analyzing medications. '
        'The user might provide you with some assisting question-answer pairs as context. You should check whether the information within these pairs is related to the user\'s query. '
        'The question-answer pairs are given within <context> and </context> tags. '
        'You should answer based on primarily your own knowledge and secondarily using the pairs that are relevant.'
    ),
}

REASONING      = True # Always ON
MAX_NEW_TOKENS = 2048
TEMPERATURE    = 0.10
TOP_P          = 0.9
SHOW_STATS     = True

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

    if st.button('🌐 General', use_container_width = True):
        reset_session()
        set_app_state(dict())
        st.rerun()

    if st.button('📝 Transcription Analysis', use_container_width = True):
        reset_session()
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['TRANSCRIPTION_CHAT'],
            MEDICAL_NER_ON = True,
            MEDICAL_RAG_ON = True,
            REDACTOR_ON    = True,
            CLS_SPEC_ON    = True,
            PROMPT_PLACEHOLDER = 'Paste your transcription here...',
            MODE_CAPTION = '📝 Transcription Analysis',
        ))
        st.rerun()

    if st.button('💬 Patient Feedback Followup', use_container_width = True):
        reset_session()
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['PATIENT_CONNECT'],
            MEDICAL_NER_ON = True,
            REDACTOR_ON    = True,
            CLS_SENT_ON    = True,
            PROMPT_PLACEHOLDER = 'Paste patient feedback message here...',
            MODE_CAPTION = '💬 Patient Feedback Followup',
        ))
        st.rerun()

    if st.button('💊 Medicine Board', use_container_width = True):
        reset_session()
        set_app_state(dict(
            SYSTEM_PROMPT = SYSTEM_PROMPTS['MEDICINE_EVALUATION'],
            MEDICAL_NER_ON = True,
            MEDICAL_RAG_ON = True,
            CLS_SPEC_ON    = True,
            PROMPT_PLACEHOLDER = 'Ask about any medicine...',
            MODE_CAPTION = '💊 Medicine Board',
        ))
        st.rerun()

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
@st.cache_resource(show_spinner='⏳ Loading the system - this may take a few moments')
def load_environment(base_model_name: str, lora_path: str):
    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(f'LoRA model not found: {lora_path_obj}')

    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

    # model = AutoModelForCausalLM.from_pretrained(
    #     llm_name,
    #     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    #     device_map = 'auto',
    # )
    # model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Updated device to add mps support
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dtype  = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        dtype  = torch.float16
    else:
        device = torch.device('cpu')
        dtype  = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype = dtype,
        trust_remote_code = True,
    ).to(device) # type: ignore

    model = PeftModel.from_pretrained(base_model, str(lora_path_obj))
    model.eval()

    med_ner = MedicalNERModule()

    med_rag = RetrieverModule()

    redactor = CensorshipModule(placeholder = '███')

    cls_specialty = SpecialtyClassifierModule('./specialty_classifier_objects.joblib')

    cls_sentiment = SentimentAnalysisModule('./sentiment_classifier_objects.joblib')

    return model, tokenizer, med_ner, med_rag, redactor, cls_specialty, cls_sentiment


model, tokenizer, med_ner, med_rag, redactor, cls_specialty, cls_sentiment = load_environment(
    BASE_MODEL_NAME,
    str(LORA_PATH)
)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    reset_session()

# ─────────────────────────────────────────────────────────────
# Streaming generator
# ─────────────────────────────────────────────────────────────
_STRIP = ['<|im_end|>', '<|endoftext|>', '<|im_start|>'] # Special tokens that may leak when skip_special_tokens=False

RESP_STATE_INIT = 0
RESP_STATE_COT  = 1
RESP_STATE_RESP = 2

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

def sentiment_tag(label: str):
    if label == 'NEGATIVE':
        sentiment_text = f'Negative'
        color = 'red'
    elif label == 'POSITIVE':
        sentiment_text = f'Positive'
        color = 'green'
    else:
        sentiment_text = f'Neutral'
        color = 'yellow'

    return f':{color}-badge[:material/star: Tone of this message - **{sentiment_text}**]\n\n\n\n'

def format_statistics(stats: dict):
    count    = stats['tokens_generated']
    duration = stats['seconds_taken']

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**3
        mem_peak  = torch.cuda.max_memory_allocated() / 1024**3
        mem_text  = f'{mem_alloc:.2f} GB ({mem_peak:.2f} GB Peak)'

        return '\n\n\n\n' + '\n'.join([
            '| Tokens Generated | Generation Time | Allocated Memory |',
            '| -------- | ------- | ------- |',
            f'| {count} | {duration:.2f} seconds | {mem_text} |'
        ])

    return '\n\n\n\n' + '\n'.join([
        '| Tokens Generated | Generation Time |',
        '| -------- | ------- |',
        f'| {count} | {duration:.2f} seconds |'
    ])

def format_rag_notification(documents: list[str]):
    if documents and len(documents) > 0:
        count = len(documents)
        return f'<span style="color:#808080">_**Fetching Relevant Documents**: {count} documents found_</span>\n\n\n\n'
    else:
        return ''

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

    inputs = tokenizer(prompt, return_tensors = 'pt').to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt                  = True,
        skip_special_tokens          = False, # skip_special_tokens=False so that <think> / </think> survive the decode
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

    # Run model.generate() in a background thread so the main thread
    # can iterate over the streamer without blocking.
    thread = Thread(target = model.generate, kwargs = gen_kwargs, daemon = True) # type: ignore
    thread.start()

    # ── State machine to parse <think>…</think> ───────────────
    buffer   = ''
    thinking = ''
    response = ''
    t_start  = 0 # index in buffer where thinking text begins
    t_end    = 0 # index in buffer where </think> begins
    state: int = RESP_STATE_INIT

    stats_token_generated: int = 0
    stats_start_timestamp: float = time()

    for chunk in streamer:
        buffer += chunk
        stats_token_generated += 1

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

        yield thinking, response, False, None

    # ── Final clean-up ─────────────────────────────────────────
    for tok in _STRIP:
        response = response.replace(tok, '')

    stats = {
        'tokens_generated': stats_token_generated,
        'seconds_taken': time() - stats_start_timestamp
    }
    yield thinking.strip(), response.strip(), True, stats

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

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):

        if msg['role'] == 'assistant' and msg.get('thinking'):
            with st.expander('💭 Thought Process', expanded = False):
                st.markdown(msg['thinking'])

        if MEDICAL_RAG_ON and msg['role'] == 'user':
            msg['content'] = med_rag.strip_rag_docs_for_display(msg['content'])

        if msg['role'] == 'user':
            user_content = msg['content']
            # specialty
            if 'rendered_ner' in msg:
                user_content = msg['rendered_ner']

            if 'specialty' in msg:
                user_content = specialty_tag(msg['specialty']) + user_content

            if 'sentiment' in msg:
                user_content = sentiment_tag(msg['sentiment']) + user_content

            st.markdown(user_content, unsafe_allow_html = True)
        elif msg['role'] == 'assistant' and msg.get('stats'):
            statistics = msg.get('stats')
            st.markdown(msg['content'] + format_statistics(statistics), unsafe_allow_html = True)
        else:
            st.markdown(msg['content'], unsafe_allow_html = True)

# Accept new user input
if user_input := st.chat_input(PROMPT_PLACEHOLDER):

    documents: list[str] = []

    with st.spinner('Working on your request...'):
        spec_added = False
        if CLS_SPEC_ON:
            if 'spec_detected' not in st.session_state:
                st.session_state['spec_detected'] = cls_specialty(user_input)['prediction']
                spec_added = True

        sent_added = False
        if CLS_SENT_ON:
            if 'sent_detected' not in st.session_state:
                st.session_state['sent_detected'] = cls_sentiment(user_input)[0]
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
            snt_label = st.session_state['sent_detected']
            user_input_for_display = sentiment_tag(snt_label) + user_input_for_display
            new_user_message['sentiment'] = st.session_state['sent_detected']

        st.markdown(user_input_for_display, unsafe_allow_html = True)

    # ── Stream assistant response ─────────────────────────────
    with st.chat_message('assistant'):
        # Thinking lives inside a collapsible expander (starts open while streaming)
        think_exp  = st.expander('💭 Thinking...', expanded = True)
        think_slot = think_exp.empty()   # updated token-by-token
        resp_slot  = st.empty()          # updated token-by-token

        last_thinking = ''
        last_response = ''

        message_history = st.session_state.messages

        statistics = None

        for thinking, response, is_final, stats in stream_response(
            history    = message_history,
            system     = SYSTEM_PROMPT,
            think      = REASONING,
            max_tokens = MAX_NEW_TOKENS,
            temp       = TEMPERATURE,
            tp         = TOP_P,
        ):
            last_thinking = thinking
            last_response = response
            statistics    = stats

            if thinking:
                think_slot.markdown(format_rag_notification(documents) + thinking, unsafe_allow_html = True)

            # Append a blinking cursor while still streaming
            resp_slot.markdown(response if is_final else response + ' ▌')

        # Finalize display (remove cursor, freeze thinking)
        think_slot.markdown(format_rag_notification(documents) + last_thinking, unsafe_allow_html = True)
        resp_slot.markdown(last_response + (format_statistics(statistics) if statistics is not None else ''))

    # ── Persist to history ────────────────────────────────────
    st.session_state.messages.append({
        'role':     'assistant',
        'content':  last_response,
        'thinking': last_thinking,
        'stats':    statistics
    })
