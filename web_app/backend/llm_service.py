from pathlib import Path 
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from peft import PeftModel 

ROOT_DIR = Path(__file__).resolve().parents[2]

BASE_MODEL_NAME = "microsoft/biogpt"
LORA_PATH = ROOT_DIR / "LLM-BioGPT" / "biogpt-familydoctor-final"


if not LORA_PATH.exists():
    raise FileNotFoundError(
        f"LoRA model not found: {LORA_PATH}"
    )
_tokenizer = None 
_model = None 

def load_model():
    global _tokenizer, _model 

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model 
    
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        trust_remote_code = True
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map = "auto" if torch.cuda.is_available() else None, 
        trust_remote_code = True
    )

    base_model.config.tie_word_embeddings = False 

    _model = PeftModel.from_pretrained(
        base_model, 
        str(LORA_PATH)
    )

    _model.eval() 

    return _tokenizer, _model 

def generate_answer(question: str, max_new_tokens: int = 250) -> str:
    tokenizer, model = load_model() 

    system_prompt = ( 
        "You are a helpful medical assistant. "
        "Answer the user's question concisely and accurately. "
        "This is for educational purposes only, not medical advice."
    )

    prompt = f"{system_prompt}\nUser: {question}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = max_new_tokens, 
            do_sample = False, 
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 4,
            pad_token_id = tokenizer.pad_token_id, 
            eos_token_id = tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens = True)

    if "Assistant:" in full_output:
        answer = full_output.split("Assistant:")[-1]
    else:
        answer = full_output
    
    return answer.strip()