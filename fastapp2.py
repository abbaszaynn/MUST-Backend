import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F  # Fixed missing import
import pickle
import numpy as np
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import shap
import lime
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def clean_text(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[\[\]{}()<>]', ' ', text)  
    text = text.lower()
    return text.strip()

def is_valid_input(text):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 2:
        return False
    if re.fullmatch(r'\d+', text):
        return False
    if re.fullmatch(r'[^a-zA-Zا-ی0-9]+', text):
        return False
    if re.fullmatch(r'[\[\](){}<>]+', text):
        return False
    if re.search(r'<[^>]*>', text):
        return False
    if not re.search(r'[a-zA-Zا-ی]', text):
        return False
    total_chars = len(text)
    alpha_chars = len(re.findall(r'[a-zA-Zا-ی]', text))
    if total_chars > 0 and (alpha_chars / total_chars) < 0.3:
        return False

    return True

def detect_language(text):
    if not is_valid_input(text):
        return "Invalid"
    cleaned_text = clean_text(text)
    if len(cleaned_text) < 2:
        return "Invalid"
    try:
        x = cv.transform([cleaned_text]).toarray()
        lang = detection_model.predict(x)
        detected_language = le[lang[0]]
        if detected_language not in supported_languages:
            return "Invalid"

        return detected_language
    except Exception as e:
        print(f"Error in language detection: {str(e)}")
        return "Invalid"


class XAITextClassifier:
    def __init__(self, model, tokenizer, device, label_mapping):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_mapping = label_mapping

        self.lime_explainer = LimeTextExplainer(
            class_names=list(label_mapping.values()),
        )

        try:
            self.shap_explainer = shap.Explainer(self.predict_proba, self.tokenizer)
        except Exception as e:
            print(f"SHAP initialization failed: {e}")
            self.shap_explainer = None

    def predict_proba(self, texts):
        """Prediction function for LIME and SHAP"""
        if isinstance(texts, str):
            texts = [texts]

        probabilities = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=128
            )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                probabilities.append(probs)
        return np.array(probabilities)

    def get_lime_explanation(self, text, num_features=10):
        try:
            explanation = self.lime_explainer.explain_instance(
                text,
                self.predict_proba,
                num_features=num_features,
                labels=[1, 2]  
            )
            return explanation
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None

    def get_shap_explanation(self, text):
        """Get SHAP explanation for the text"""
        if self.shap_explainer is None:
            return None    
        try:
            shap_values = self.shap_explainer([text])
            return shap_values
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None

    def extract_harmful_words(self, text, threshold=0.1):
        harmful_words = set()
        try:
            lime_exp = self.get_lime_explanation(text)
            if lime_exp is None:
                return harmful_words
            for label in [1, 2]:
                if lime_exp.available_labels() and label in lime_exp.available_labels():
                    exp_list = lime_exp.as_list(label=label)
                    for word, score in exp_list:
                        if score > threshold:
                            clean_word = word.lower().strip('.,!?;:"()[]{}')
                            if clean_word: 
                                harmful_words.add(clean_word)
        except Exception as e:
            print(f"LIME analysis failed during harmful word extraction: {e}")
            return set() 

        return harmful_words

    def modify_text(self, text, harmful_words):
        words = text.split()
        modified_words = []

        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            if clean_word not in harmful_words:
                modified_words.append(word)
            else:
                modified_words.append('[REMOVED]')

        if not modified_words or all(w == '[REMOVED]' for w in modified_words):
             return "" 

        return ' '.join(modified_words)

    def classify_with_explanation(self, text):
        try:
            probabilities = self.predict_proba(text)
            if probabilities.shape[0] == 0:
                return "Error"

            probabilities = probabilities[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.label_mapping[predicted_class_idx]

            print(f"\nORIGINAL TEXT:")
            print(f"   {text}")
            print(f"\nPREDICTED CLASS: {predicted_class.upper()}")
            print(f"\nCLASS PROBABILITIES:")
            for idx, prob in enumerate(probabilities):
                label = self.label_mapping[idx]
                print(f"   {label.capitalize()}: {prob:.4f} ({prob * 100:.2f}%)")

        except Exception as e:
            return "Error"

        if predicted_class_idx in [1, 2]: 
            print(f"\nXAI ANALYSIS (Text classified as {predicted_class}):")

            harmful_words = self.extract_harmful_words(text)

            if harmful_words:
                print(f"\nHARMFUL WORDS IDENTIFIED:")
                print(f"   {', '.join(sorted(harmful_words))}")

                modified_text = self.modify_text(text, harmful_words)
                print(f"\nMODIFIED TEXT (harmful words removed/marked):")
                if modified_text:
                    print(f"   {modified_text}")
                else:
                    print("   [All words identified as harmful or text became empty]")

        return predicted_class
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model_path = str((BASE_DIR / "roberta_combine_f1").resolve())
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
label_mapping_inverse = {0: 'neutral', 1: 'offensive', 2: 'hate'}

with open(BASE_DIR / "detection_model.pkl", "rb") as f:
    detection_model = pickle.load(f)
with open(BASE_DIR / "transform.pkl", "rb") as f:
    cv = pickle.load(f)
le = np.load(BASE_DIR / "classes.npy", allow_pickle=True)

supported_languages = {"English", "Urdu", "Roman urdu"} 


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_2.html", {"request": request})

@app.post("/analyze")
async def analyze(data: dict):
    try:
        text = data.get("text", "").strip()

        if not text:
            return JSONResponse(content={
                "error": True,
                "message": "No text provided. Please enter a valid sentence."
            }, status_code=400)

        if not is_valid_input(text):
            return JSONResponse(content={
                "error": True,
                "message": "Invalid input. Text must contain words in English, Urdu, or Roman Urdu."
            }, status_code=400)

        detected_lang = detect_language(text)

        if detected_lang == "Invalid" or detected_lang not in supported_languages:
            return JSONResponse(content={
                "error": True,
                "message": "Invalid input. Please enter text in English, Urdu, or Roman Urdu."
            }, status_code=400)
        classifier = XAITextClassifier(model, tokenizer, device, label_mapping_inverse)
        predicted_class = classifier.classify_with_explanation(text)
        probabilities = classifier.predict_proba(text)[0]
        scores = {
            "neutral": float(round(float(probabilities[0]) * 100, 2)),
            "offensive": float(round(float(probabilities[1]) * 100, 2)),
            "hate": float(round(float(probabilities[2]) * 100, 2)),
        }
        confidence = float(round(float(probabilities[np.argmax(probabilities)]) * 100, 1))

        removed_words = []
        cleaned_text = text
        word_contributions = []

        if predicted_class in ['hate', 'offensive']:
            harmful_words = classifier.extract_harmful_words(text)

            if harmful_words:
                removed_words = sorted(list(harmful_words))
                cleaned_text = classifier.modify_text(text, harmful_words)

                lime_exp = classifier.get_lime_explanation(text)
                if lime_exp:
                    label_idx = 1 if predicted_class == 'offensive' else 2
                    if label_idx in lime_exp.available_labels():
                        word_contributions = lime_exp.as_list(label=label_idx)

        response_content = {
            "error": False,
            "original_text": text,
            "text": text,
            "cleaned_text": cleaned_text,
            "removed_words": removed_words,
            "category": predicted_class,
            "language": detected_lang,
            "confidence": confidence,
            "scores": scores
        }

        if word_contributions:
            response_content["word_contributions"] = [
                {"word": word, "contribution": float(round(float(contrib), 4))}
                for word, contrib in word_contributions[:15]
            ]
        
        # Log to Database
        log_analysis(text, predicted_class, confidence, detected_lang)

        return JSONResponse(content=response_content)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(content={
            "error": True,
            "message": f"Error: {str(e)}"
        }, status_code=500)

# --- Database Setup ---
import sqlite3
from datetime import datetime

DB_NAME = str(BASE_DIR / "hatespeech.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Logs table
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  category TEXT,
                  confidence REAL,
                  language TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    # Users table (for monitoring)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  platform TEXT,
                  risk_score REAL,
                  last_active DATETIME)''')
    
    # Seed some dummy users if empty
    c.execute("SELECT count(*) FROM users")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO users (username, platform, risk_score, last_active) VALUES (?, ?, ?, ?)",
                      [('user123', 'Facebook', 85.5, datetime.now()),
                       ('cool_guy', 'Twitter', 12.0, datetime.now()),
                       ('angry_bird', 'Facebook', 92.1, datetime.now())])
    
    conn.commit()
    conn.close()

init_db()


def _rows_to_dicts(cursor):
    rows = cursor.fetchall()
    out = []
    for row in rows:
        out.append({k: row[k] for k in row.keys()})
    return out


def log_analysis(text, category, confidence, language):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO logs (text, category, confidence, language) VALUES (?, ?, ?, ?)",
                  (text, category, confidence, language))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# --- New Endpoints ---

@app.post("/scrape")
async def scrape_profile(data: dict):
    """
    Mock endpoint to simulate scraping a profile.
    Real scraping is complex and often blocked.
    """
    profile_url = data.get("url", "")
    if not profile_url:
         return JSONResponse(content={"error": True, "message": "URL required"}, status_code=400)
    
    # Mock data generation
    posts = []
    username = profile_url.split("/")[-1] or "unknown_user"
    
    # Simulate 5 posts
    sample_texts = [
        "I love this beautiful day!",
        "These people are ruining our country, they should be kicked out.",
        "Just had a great meal.",
        "Why are they so stupid? I hate them.",
        "Working hard on my new project."
    ]
    
    for _ in range(5):
        text = random.choice(sample_texts)
        # Analyze the mock text using our internal function logic (simplified here for speed, or call analyze internally)
        # For the mock, we'll just randomly assign categories based on text content roughly
        if "hate" in text or "kicked" in text:
            cat = "hate"
            conf = random.uniform(80, 99)
        elif "stupid" in text:
            cat = "offensive"
            conf = random.uniform(60, 80)
        else:
            cat = "neutral"
            conf = random.uniform(70, 99)
            
        posts.append({
            "content": text,
            "category": cat,
            "confidence": round(conf, 1),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        
        # Log to DB
        log_analysis(text, cat, conf, "English")

    return JSONResponse(content={
        "error": False,
        "username": username,
        "posts": posts,
        "risk_score": random.randint(0, 100) # Mock risk score
    })

@app.get("/trends")
async def get_trends():
    """Aggregate logs for trend analysis"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # Get counts by category
        c.execute("SELECT category, COUNT(*) FROM logs GROUP BY category")
        counts = dict(c.fetchall())
        
        # Get recent activity (last 10 logs)
        c.execute("SELECT timestamp, category FROM logs ORDER BY id DESC LIMIT 20")
        recent = [{"time": row[0], "category": row[1]} for row in c.fetchall()]
        
        conn.close()
        
        return JSONResponse(content={
            "error": False,
            "stats": {
                "hate": counts.get("hate", 0),
                "offensive": counts.get("offensive", 0),
                "neutral": counts.get("neutral", 0),
                "total": sum(counts.values())
            },
            "recent_activity": recent
        })
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)

@app.get("/flagged")
async def get_flagged():
    """Get recent hate/offensive content"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM logs WHERE category IN ('hate', 'offensive') ORDER BY id DESC LIMIT 50")
        rows = _rows_to_dicts(c)
        conn.close()
        return JSONResponse(content={"error": False, "data": rows})
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)


@app.get("/live-feed")
async def get_live_feed():
    """Recent flagged items formatted for the dashboard live feed (includes display fields)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM logs WHERE category IN ('hate', 'offensive') ORDER BY id DESC LIMIT 30"
        )
        rows = _rows_to_dicts(c)
        conn.close()
        platforms = ["Facebook", "Twitter", "Web"]
        for row in rows:
            row.setdefault("username", f"user_{row.get('id', 'anon')}")
            row.setdefault("platform", random.choice(platforms))
            if row.get("timestamp") is not None:
                row["timestamp"] = str(row["timestamp"])
            if row.get("confidence") is not None:
                row["confidence"] = float(row["confidence"])
        return JSONResponse(content={"error": False, "data": rows})
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)


@app.get("/monitoring")
async def get_monitoring():
    """Get monitored users"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users ORDER BY risk_score DESC")
        rows = _rows_to_dicts(c)
        conn.close()
        return JSONResponse(content={"error": False, "data": rows})
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)

