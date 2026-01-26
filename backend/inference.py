import os
import tensorflow as tf
import numpy as np
import pickle
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import MarianMTModel, MarianTokenizer

# ==========================================
# CONFIGURATION & PATHS
# ==========================================

# Base paths (Assuming this script is in /backend, so models are in ../)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # e:\CCP2_TAMtoENG_AAC\tamil_next_word_app\new ui

# Model filenames
BI_GRU_PATH = os.path.join(PROJECT_ROOT, "bi_gru_1111.keras")
LSTM_PATH = os.path.join(PROJECT_ROOT, "baseline_lstm_glove.keras")
BILSTM_PATH = os.path.join(PROJECT_ROOT, "bilstm_glove_finetuned.keras")

# Tokenizer filenames
WORD2IDX_PATH = os.path.join(PROJECT_ROOT, "word2idxlarge.pkl")
IDX2WORD_PATH = os.path.join(PROJECT_ROOT, "idx2wordlarge.pkl")
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
try:
    tokenizer_ta_en = MarianTokenizer.from_pretrained(TA_EN_DIR, local_files_only=True)
    model_ta_en = MarianMTModel.from_pretrained(TA_EN_DIR, local_files_only=True).to(DEVICE)

    tokenizer_en_ta = MarianTokenizer.from_pretrained(EN_TA_DIR, local_files_only=True)
    model_en_ta = MarianMTModel.from_pretrained(EN_TA_DIR, local_files_only=True).to(DEVICE)
    print("✅ NMT Models loaded.")
except Exception as e:
    print(f"❌ Error loading NMT models: {e}")
    # We might want to continue if NMT fails, but for this app it's critical
    raise e


# ==========================================
# HELPER FUNCTIONS
# ==========================================

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


def predict_tamil_bigru(text, top_k=3):
    tokens = text.strip().split()
    encoded = [ta_word2idx.get(w, ta_word2idx.get(UNK, 0)) for w in tokens]
    # Pad sequences
    padded = pad_sequences([encoded], maxlen=MAX_SEQ_LEN_TA, padding="pre")

    # Predict
    probs = bi_gru_model.predict(padded, verbose=0)[0]
    
    # Get Top-K
    top_idx = np.argsort(probs)[-top_k:][::-1]
    
    results = []
    for i in top_idx:
        word = ta_idx2word.get(i, "")
        confidence = float(probs[i])
        results.append({"word": word, "confidence": confidence})
        
    top1_prob = float(probs[top_idx[0]])
    return results, top1_prob


def predict_english(model, tokenizer, text, top_k=3):
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


def translation_pipeline_predict(text_ta, model, tokenizer, top_k=3):
    try:
        # 1. Tamil -> English
        text_en = translate_offline(text_ta, direction="ta_en")
        
        # 2. Predict next English words
        en_preds, top1_prob = predict_english(model, tokenizer, text_en, top_k)
        
        # 3. English -> Tamil (Back translate predictions)
        ta_preds = []
        for en_word, prob in en_preds:
            ta_word = translate_offline(en_word, direction="en_ta")
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
    sentence_tokens = current_sentence.strip().split()
    if not sentence_tokens:
        return True # Can't fail if no context
        
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
                # Check specifics to avoid false positives on 'valid' mismatched verbs
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


def get_predictions(current_sentence, language="ta"):
    """
    Main entry point for API.
    Returns dictionary with predictions for all 3 models.
    """
    response = {}
    
    if language == "en":
        # ENGLISH TAB: Use LSTM/BiLSTM directly, NO Bi-GRU
        
        # 1. Bi-GRU (Not used for English)
        response["bi_gru"] = {"predictions": [], "top1_confidence": 0.0, "error": "Not applicable for English"}

        # 2. LSTM (Direct English)
        try:
            preds, conf = predict_english(lstm_model, en_tokenizer_lstm, current_sentence)
            # English Validity Check
            # For now, reusing calculate_validity_score which applies Tamil rules. 
            # We need English rules. But for Step 1, let's just get predictions.
            # TODO: Implement english specific validity if needed or adapt existing one.
            # The user asked for "Graph 2 — Grammatical Validity Rate ... Rule-based English grammar checks"
            # We will use a modified validity checker or just existing one if it's generic enough (it isn't).
            # For now, let's map it to 0 or implement basic English validity.
            # Let's add a simple English validity check helper or update the existing one.
            # Actually, let's just use the calculate_validity_score but we need to ensure it doesn't fail.
            # The current is_grammatically_valid is heavily Tamil specific.
            # We should probably pass language to calculate_validity_score too.
            validity = calculate_validity_score(preds, current_sentence, language="en")
            response["lstm"] = {
                "predictions": preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"LSTM Error: {e}")
            response["lstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 3. BiLSTM (Direct English)
        try:
            preds, conf = predict_english(bilstm_model, en_tokenizer_bilstm, current_sentence)
            validity = calculate_validity_score(preds, current_sentence, language="en")
            response["bilstm"] = {
                "predictions": preds,
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
            validity = calculate_validity_score(preds, current_sentence, language="ta")
            response["bi_gru"] = {
                "predictions": preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"Bi-GRU Error: {e}")
            response["bi_gru"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 2. LSTM Baseline
        try:
            preds, conf = translation_pipeline_predict(current_sentence, lstm_model, en_tokenizer_lstm)
            validity = calculate_validity_score(preds, current_sentence, language="ta")
            response["lstm"] = {
                "predictions": preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"LSTM Error: {e}")
            response["lstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

        # 3. BiLSTM Model
        try:
            preds, conf = translation_pipeline_predict(current_sentence, bilstm_model, en_tokenizer_bilstm)
            validity = calculate_validity_score(preds, current_sentence, language="ta")
            response["bilstm"] = {
                "predictions": preds,
                "top1_confidence": conf,
                "validity_score": validity
            }
        except Exception as e:
            print(f"BiLSTM Error: {e}")
            response["bilstm"] = {"predictions": [], "top1_confidence": 0.0, "error": str(e)}

    return response

if __name__ == "__main__":
    # Simple Test
    test_sent = "நான் இன்று"
    print(f"\nTesting with: '{test_sent}'")
    res = get_predictions(test_sent)
    print(res)
