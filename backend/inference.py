import os
import tensorflow as tf
import numpy as np
import pickle
import torch
import asyncio
import functools
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import MarianMTModel, MarianTokenizer

# ==========================================
# CONFIGURATION & PATHS
# ==========================================

# Base paths (Assuming this script is in /backend, so models are in ../)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # e:\CCP2_TAMtoENG_AAC\tamil_next_word_app\new ui

# Model filenames
BI_GRU_PATH = os.path.join(PROJECT_ROOT, "bi_gru_1500.keras")
LSTM_PATH = os.path.join(PROJECT_ROOT, "baseline_lstm_glove.keras")
BILSTM_PATH = os.path.join(PROJECT_ROOT, "bilstm_glove_finetuned.keras")

# Tokenizer filenames
WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "word2idx1500.pkl")
IDX2WORD_PATH = os.path.join(PROJECT_ROOT, "idx2word1500.pkl")
TOKENIZER_LSTM_PATH = os.path.join(PROJECT_ROOT, "tokenizer_baseline.pkl")
TOKENIZER_BILSTM_PATH = os.path.join(PROJECT_ROOT, "tokenizer_bilstm_glove.pkl")

# NMT Paths
TA_EN_DIR = os.path.join(PROJECT_ROOT, "saved_nmt", "tamil_to_english")
EN_TA_DIR = os.path.join(PROJECT_ROOT, "saved_nmt", "english_to_tamil")

# Constants
MAX_SEQ_LEN_TA = 12
MAX_SEQ_LEN_EN = 15
UNK = "<UNK>"

# Device for Transformers
DEVICE = "cpu"

# ==========================================
# GLOBAL LOADED MODELS
# ==========================================
print("⏳ Loading models... This may take a few seconds.")

# 1. Load Keras Models
try:
    bi_gru_model = tf.keras.models.load_model(BI_GRU_PATH, compile=False)
    lstm_model = tf.keras.models.load_model(LSTM_PATH, compile=False)
    bilstm_model = tf.keras.models.load_model(BILSTM_PATH, compile=False)
    print("✅ Keras models loaded.")
except Exception as e:
    print(f"❌ Error loading Keras models: {e}")
    raise e

# 2. Load Pickles
try:
    with open(WORD2IDX_PATH, "rb") as f:
        ta_word2idx = pickle.load(f)
    with open(IDX2WORD_PATH, "rb") as f:
        ta_idx2word = pickle.load(f)
    print(f"✅ Tamil Vocab loaded (Size: {len(ta_word2idx)})")

    with open(TOKENIZER_LSTM_PATH, "rb") as f:
        en_tokenizer_lstm = pickle.load(f)
    with open(TOKENIZER_BILSTM_PATH, "rb") as f:
        en_tokenizer_bilstm = pickle.load(f)
    print("✅ English Tokenizers loaded.")
except Exception as e:
    print(f"❌ Error loading pickle files: {e}")
    raise e

# 3. Load Translation Models (Offline)
# 3. Load Translation Models
try:
    print("Trying to load NMT models locally...")
    tokenizer_ta_en = MarianTokenizer.from_pretrained(TA_EN_DIR, local_files_only=True)
    model_ta_en = MarianMTModel.from_pretrained(TA_EN_DIR, local_files_only=True).to(DEVICE)

    tokenizer_en_ta = MarianTokenizer.from_pretrained(EN_TA_DIR, local_files_only=True)
    model_en_ta = MarianMTModel.from_pretrained(EN_TA_DIR, local_files_only=True).to(DEVICE)
    print("✅ NMT Models loaded from local directory.")
except Exception as e:
    print(f"⚠️ Local NMT load failed ({e}). Attempting download from Hugging Face Hub...")
    try:
        # Fallback to Helsinki-NLP official models
        # Tamil -> English (Dravidian -> English)
        TA_EN_MODEL_ID = "Helsinki-NLP/opus-mt-dra-en"
        tokenizer_ta_en = MarianTokenizer.from_pretrained(TA_EN_MODEL_ID)
        model_ta_en = MarianMTModel.from_pretrained(TA_EN_MODEL_ID).to(DEVICE)
        
        # English -> Tamil (English -> Dravidian)
        EN_TA_MODEL_ID = "Helsinki-NLP/opus-mt-en-dra"
        tokenizer_en_ta = MarianTokenizer.from_pretrained(EN_TA_MODEL_ID)
        model_en_ta = MarianMTModel.from_pretrained(EN_TA_MODEL_ID).to(DEVICE)
        print("✅ NMT Models loaded from Hugging Face Hub.")
    except Exception as e2:
         print(f"❌ Error loading NMT models from Hub: {e2}")
         raise e2


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def convert_to_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def translate_offline(text, direction="ta_en"):
    """
    Translates text using loaded offline Helsinki-NLP models.
    """
    if not text or not text.strip():
        return ""

    if direction == "ta_en":
        inputs = tokenizer_ta_en(text, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            output = model_ta_en.generate(**inputs)
        return tokenizer_ta_en.batch_decode(output, skip_special_tokens=True)[0]
    else:
        # English to Tamil requires the >>tam<< prefix for some multi-target models, 
        # but the notebook used specific en-dra models.
        # Checking notebook logic: inputs = tokenizer_en_ta(">>tam<< " + text, ...)
        # The notebook explicitly adds ">>tam<< ", so we will too.
        inputs = tokenizer_en_ta(">>tam<< " + text, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            output = model_en_ta.generate(**inputs)
        return tokenizer_en_ta.batch_decode(output, skip_special_tokens=True)[0]

@functools.lru_cache(maxsize=512)
def translate_cached(text, direction):
    """Cached version of translate_offline for frequent predictions."""
    return translate_offline(text, direction)


def filter_predictions_by_prefix(predictions, prefix, top_k=3):
    """
    Filter predictions to only include words starting with the given prefix.
    Used for AAC-style prefix-based word completion.
    
    Args:
        predictions: List of {"word": str, "confidence": float} dicts
        prefix: String prefix to filter by (case-insensitive)
        top_k: Number of results to return
    
    Returns:
        Filtered list of top_k predictions matching prefix
    """
    if not prefix or not prefix.strip():
        # No prefix, return top predictions as-is
        return predictions[:top_k]
    
    prefix_lower = prefix.lower().strip()
    
    # Filter predictions that start with the prefix
    filtered = [
        p for p in predictions 
        if p.get('word', '').lower().startswith(prefix_lower)
    ]
    
    # Return top_k filtered results
    return filtered[:top_k]


def predict_tamil_bigru(text, top_k=30):
    """
    Predict next Tamil words using Bi-GRU model.
    Returns more candidates (top_k=30) to allow for prefix filtering.
    """
    tokens = text.strip().split()
    encoded = [ta_word2idx.get(w, ta_word2idx.get(UNK, 0)) for w in tokens]
    # Pad sequences
    padded = pad_sequences([encoded], maxlen=MAX_SEQ_LEN_TA, padding="pre")

    # Predict
    probs = bi_gru_model.predict(padded, verbose=0)[0]
    
    # Get Top-K (increased to allow filtering)
    top_idx = np.argsort(probs)[-top_k:][::-1]
    
    results = []
    for i in top_idx:
        word = ta_idx2word.get(i, "")
        confidence = float(probs[i])
        results.append({"word": word, "confidence": confidence})
        
    top1_prob = float(probs[top_idx[0]]) if len(top_idx) > 0 else 0.0
    return results, top1_prob


def predict_english(model, tokenizer, text, top_k=30):
    """
    Predict next English words using LSTM/BiLSTM model.
    Returns more candidates (top_k=30) to allow for prefix filtering.
    """
    if not text or not text.strip():
        return [], 0.0

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN_EN, padding="pre")

    probs = model.predict(padded, verbose=0)[0]
    top_idx = np.argsort(probs)[-top_k:][::-1]

    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    
    results = []
    for i in top_idx:
        word = inv_vocab.get(i, "")
        confidence = float(probs[i])
        results.append({"word": word, "confidence": confidence})
        
    top1_prob = float(probs[top_idx[0]]) if len(top_idx) > 0 else 0.0
    return results, top1_prob


def translate_batch(words, direction="en_ta"):
    """
    Batch translate multiple words at once for better performance.
    """
    if not words:
        return []
    
    # Filter empty words
    words = [w for w in words if w and w.strip()]
    if not words:
        return []
    
    try:
        if direction == "en_ta":
            # Batch translate English words to Tamil
            inputs = tokenizer_en_ta([">>tam<< " + w for w in words], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model_en_ta.generate(**inputs)
            return tokenizer_en_ta.batch_decode(outputs, skip_special_tokens=True)
        else:
            # Batch translate Tamil to English
            inputs = tokenizer_ta_en(words, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model_ta_en.generate(**inputs)
            return tokenizer_ta_en.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        print(f"Batch translation error: {e}")
        return [translate_offline(w, direction) for w in words]

def translation_pipeline_predict(text_ta, model, tokenizer, top_k=3):
    try:
        # 1. Tamil -> English
        text_en = translate_cached(text_ta, direction="ta_en")
        
        # 2. Predict next English words
        en_preds, top1_prob = predict_english(model, tokenizer, text_en, top_k)
        
        # 3. Batch English -> Tamil (Translate all predictions at once)
        en_words = [item["word"] for item in en_preds]
        ta_words = translate_batch(en_words, direction="en_ta")
        
        ta_preds = []
        for i, en_item in enumerate(en_preds):
            ta_word = ta_words[i] if i < len(ta_words) else ""
            prob = en_item["confidence"]
            ta_preds.append({"word": ta_word, "confidence": prob})
            
        return ta_preds, top1_prob
        
    except Exception as e:
        print(f"Error in translation pipeline: {e}")
        return [], 0.0


# ==========================================
# GRAMMATICAL RULES & VALIDATION
# ==========================================

# Heuristic Lists (Expanded)
VERBS_LIST = {
    "நடந்தான்", "வந்தான்", "பார்த்தேன்", "கேட்டேன்", "சென்றார்", "முடிந்தது", "இருக்கிறார்",
    "சாப்பிட்டேன்", "தூங்கினேன்", "பேசினேன்", "சிரித்தான்", "அழுதான்", "ஓடினான்",
    "செய்தான்", "கொடுத்தார்", "எடுத்தார்", "வைத்தார்", "பறந்தது", "கடித்தது",
    "இருந்தது", "ஆகும்", "உள்ளது", "வேண்டும்", "கூடும்", "போனான்", "நின்றான்"
}

# Common Tamil Verb Suffixes (to catch verbs not in the list)
VERB_SUFFIXES = (
    "தேன்", "னேன்", "கிறேன்", # 1st Person Sing
    "தோம்", "னோம்", "கிறோம்", # 1st Person Plural
    "தாய்", "னாய்", "கிறாய்", # 2nd Person Sing
    "தீர்கள்", "னீர்கள்", "கிறீர்கள்", # 2nd Person Plural
    "தான்", "னான்", "கிறான்", # 3rd Person Male 
    "தாள்", "னாள்", "கிறாள்", # 3rd Person Female
    "தார்", "னார்", "கிறார்", # 3rd Person Respect
    "தார்கள்", "னார்கள்", "கிறார்கள்", # 3rd Person Plural
    "தது", "னது", "கின்றது", # 3rd Person Neuter Sing
    "தன", "ின", "கின்றன"     # 3rd Person Neuter Plural
)

POSTPOSITIONS_LIST = {
    "உடன்", "மூலம்", "பற்றி", "க்காக", "இடம்", "விட", "பால்", "மேல்", "கீழ்",
    "முன்", "பின்", "உள்ளே", "வெளியே", "மீது", "வழியாக", "நடுவில்",
    "கொண்டு", "இருந்து", "வரை"
}

def is_verb(word):
    if word in VERBS_LIST:
        return True
    if word.endswith(VERB_SUFFIXES):
        return True
    return False

def is_grammatically_valid(current_sentence, next_word, language="ta"):
    """
    Rule-based validity check for a single next-word prediction.
    """
    if not current_sentence or not next_word:
        return False
    
    sentence_tokens = current_sentence.strip().split()
    if not sentence_tokens:
        return True
    
    prev_word = sentence_tokens[-1]
    
    if language == "en":
        # ENGLISH RULES
        # Rule 1: No repeated word back-to-back
        if next_word.lower() == prev_word.lower():
            return False
            
        # Rule 2: Basic English Check
        # Ensure word is English (ASCII)
        if not all(ord(c) < 128 for c in next_word):
             return False
             
        # Additional English rules can be added here
        return True

    # TAMIL RULES (Existing)
    # Rule 0: Sanity Check (No English/Numbers unless strictly allowed)
    # Check if word contains ASCII characters (heuristic for English leakage)
    if any(ord(c) < 128 for c in next_word) and not next_word.isdigit():
        # Allow numbers, but reject mixed English-Tamil or pure English
        # Exception: Punctuation (usually ASCII). Let's filter alnum.
        english_chars = [c for c in next_word if c.isalpha() and ord(c) < 128]
        if english_chars:
            return False

    # Rule 1: No repeated word back-to-back
    if next_word == prev_word:
        return False
        
    # Rule 2: No consecutive verbs
    if is_verb(prev_word) and is_verb(next_word):
        return False
        
    # Rule 3: Vocab check (STRICT ENFORCEMENT)
    # User Requirement: "Predicted word must exist in model vocabulary"
    # This penalizes Translation models if they produce OOV words, but that's the rule.
    if next_word not in ta_word2idx and next_word != UNK:
        return False

    # Rule 4: Sentence must not end with a standalone postposition
    if next_word in POSTPOSITIONS_LIST:
        return False

    # Rule 5: Subject-Verb Agreement (Restricted to I/We for demo)
    if is_verb(next_word):
        if "நான்" in sentence_tokens:
            if not (next_word.endswith("ேன்") or next_word.endswith("கிறேன்")):
                return False
        if "நாங்கள்" in sentence_tokens or "நாம்" in sentence_tokens:
             if not (next_word.endswith("ோம்") or next_word.endswith("கிறோம்")):
                return False

    return True

def calculate_validity_score(predictions, current_sentence, language="ta"):
    """
    Calculates percentage of valid predictions in the top-K set.
    """
    if not predictions:
        return 0.0
        
    valid_count = 0
    for item in predictions:
        word = item["word"] if isinstance(item, dict) else item[0] # Handle dict or tuple
        if is_grammatically_valid(current_sentence, word, language):
            valid_count += 1
            
    return (valid_count / len(predictions)) * 100


async def predict_bigru_async(current_sentence, partial_word):
    """Async wrapper for Bi-GRU prediction."""
    try:
        preds, conf = predict_tamil_bigru(current_sentence)
        filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
        validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
        return {
            "predictions": filtered_preds,
            "top1_confidence": conf,
            "validity_score": validity
        }
    except Exception as e:
        print(f"Bi-GRU Error: {e}")
        return {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

async def predict_lstm_ta_async(current_sentence, partial_word):
    """Async wrapper for LSTM (Tamil pipeline) prediction."""
    try:
        preds, conf = translation_pipeline_predict(current_sentence, lstm_model, en_tokenizer_lstm)
        filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
        validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
        return {
            "predictions": filtered_preds,
            "top1_confidence": conf,
            "validity_score": validity
        }
    except Exception as e:
        print(f"LSTM Error: {e}")
        return {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

async def predict_bilstm_ta_async(current_sentence, partial_word):
    """Async wrapper for BiLSTM (Tamil pipeline) prediction."""
    try:
        preds, conf = translation_pipeline_predict(current_sentence, bilstm_model, en_tokenizer_bilstm)
        filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
        validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
        return {
            "predictions": filtered_preds,
            "top1_confidence": conf,
            "validity_score": validity
        }
    except Exception as e:
        print(f"BiLSTM Error: {e}")
        return {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

async def predict_lstm_en_async(current_sentence, partial_word):
    """Async wrapper for LSTM (English) prediction."""
    try:
        preds, conf = predict_english(lstm_model, en_tokenizer_lstm, current_sentence)
        filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
        validity = calculate_validity_score(filtered_preds, current_sentence, language="en")
        return {
            "predictions": filtered_preds,
            "top1_confidence": conf,
            "validity_score": validity
        }
    except Exception as e:
        print(f"LSTM Error: {e}")
        return {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

async def predict_bilstm_en_async(current_sentence, partial_word):
    """Async wrapper for BiLSTM (English) prediction."""
    try:
        preds, conf = predict_english(bilstm_model, en_tokenizer_bilstm, current_sentence)
        filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
        validity = calculate_validity_score(filtered_preds, current_sentence, language="en")
        return {
            "predictions": filtered_preds,
            "top1_confidence": conf,
            "validity_score": validity
        }
    except Exception as e:
        print(f"BiLSTM Error: {e}")
        return {"predictions": [], "top1_confidence": 0.0, "error": str(e)}


def get_predictions(current_sentence, language="ta", partial_word=""):
    """
    Main entry point for API.
    Returns dictionary with predictions for all 3 models.
    Runs model inference in parallel for better performance.
    
    Args:
        current_sentence: The complete sentence context
        language: "ta" for Tamil or "en" for English
        partial_word: The word currently being typed (for prefix filtering)
    """
    response = {}
    
    if language == "en":
        # ENGLISH TAB: Use LSTM/BiLSTM directly, NO Bi-GRU
        
        # 1. Bi-GRU (Not used for English)
        response["bi_gru"] = {"predictions": [], "top1_confidence": 0.0, "error": "Not applicable for English"}

        # 2. LSTM (Direct English)
        try:
            preds, conf = predict_english(lstm_model, en_tokenizer_lstm, current_sentence)
            # Apply prefix filtering
            filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
            validity = calculate_validity_score(filtered_preds, current_sentence, language="en")
            response["lstm"] = {
                "predictions": filtered_preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"LSTM Error: {e}")
            response["lstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 3. BiLSTM (Direct English)
        try:
            preds, conf = predict_english(bilstm_model, en_tokenizer_bilstm, current_sentence)
            # Apply prefix filtering
            filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
            validity = calculate_validity_score(filtered_preds, current_sentence, language="en")
            response["bilstm"] = {
                "predictions": filtered_preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"BiLSTM Error: {e}")
            response["bilstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

    else:
        # TAMIL TAB: Existing Logic
    
        # 1. Direct Tamil Bi-GRU
        try:
            preds, conf = predict_tamil_bigru(current_sentence)
            # Apply prefix filtering
            filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
            validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
            response["bi_gru"] = {
                "predictions": filtered_preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"Bi-GRU Error: {e}")
            response["bi_gru"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 2. LSTM Baseline
        try:
            preds, conf = translation_pipeline_predict(current_sentence, lstm_model, en_tokenizer_lstm)
            # Apply prefix filtering
            filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
            validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
            response["lstm"] = {
                "predictions": filtered_preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"LSTM Error: {e}")
            response["lstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 3. BiLSTM Model
        try:
            preds, conf = translation_pipeline_predict(current_sentence, bilstm_model, en_tokenizer_bilstm)
            # Apply prefix filtering
            filtered_preds = filter_predictions_by_prefix(preds, partial_word, top_k=3)
            validity = calculate_validity_score(filtered_preds, current_sentence, language="ta")
            response["bilstm"] = {
                "predictions": filtered_preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"BiLSTM Error: {e}")
            response["bilstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

    return convert_to_serializable(response)

if __name__ == "__main__":
    # Simple Test
    test_sent = "நான் இன்று"
    print(f"\nTesting with: '{test_sent}'")
    res = get_predictions(test_sent)
    print(res)
