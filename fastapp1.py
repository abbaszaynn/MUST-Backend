import os
import torch
import pickle
import numpy as np
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Configure logging
logging.basicConfig(
    filename='backend.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load Model and Tokenizer
try:
    logging.info("Loading model and tokenizer...")
    model_path = os.path.join(os.getcwd(), 'roberta_combine_f1') 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # If local model fails, maybe try the huggingface one as fallback?
    # But let's stick to what was working.
    raise e

# Load other pickles
try:
    detection_model = pickle.load(open("detection_model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))
    le = np.load("classes.npy", allow_pickle=True)
except Exception as e:
    logging.error(f"Error loading pickles: {e}")
    raise e

supported_languages = {"English", "Urdu", "Roman urdu"}  
label_mapping = {0: 'neutral', 1: 'offensive', 2: 'hate'}

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index_2.html", {"request": request})

async def predict_text(text):
    """Reusable prediction logic"""
    if not text:
        return {"error": True, "message": "No text provided."}
    
    if not is_valid_input(text):
        return {"error": True, "message": "Invalid input. Text must contain words in English, Urdu, or Roman Urdu."}

    detected_lang = detect_language(text)
    if detected_lang == "Invalid" or detected_lang not in supported_languages:
        return {"error": True, "message": "Invalid input. Please enter text in English, Urdu, or Roman Urdu."}

    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).cpu().item()
    prediction = label_mapping[predicted_class]

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probabilities[0][predicted_class].cpu().item() * 100
    
    scores = {
        "neutral": round(probabilities[0][0].cpu().item() * 100, 2),
        "offensive": round(probabilities[0][1].cpu().item() * 100, 2),
        "hate": round(probabilities[0][2].cpu().item() * 100, 2),
    }
    
    return {
        "error": False,
        "category": prediction,
        "confidence": round(confidence, 1),
        "language": detected_lang,
        "scores": scores,
        "text": text
    }

@app.post("/analyze")
async def analyze(data: dict):
    try:
        text = data.get("text", "").strip()
        result = await predict_text(text)
        
        if result.get("error"):
            return JSONResponse(content=result, status_code=400)

        # Log to Database
        log_analysis(result["text"], result["category"], result["confidence"], result["language"])
        
        return JSONResponse(content={
            "error": False,
            "category": result["category"],
            "predictionProbability": result["confidence"],
            "confidence": result["confidence"],
            "explanation": f"Classified as {result['category']} with {result['confidence']}% confidence based on model analysis.",
            "language": result["language"],
            "text": result["text"],
            "scores": result["scores"]
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(content={
            "error": True,
            "message": f"Error: {str(e)}"
        }, status_code=500)

import requests

@app.post("/api/webhooks/apify")
async def apify_webhook(request: Request):
    """
    Handle incoming webhooks from Apify.
    Triggered when a scraper run finishes.
    """
    try:
        payload = await request.json()
        logging.info(f"Received Apify webhook: {payload}")
        
        # Extract dataset ID
        resource = payload.get("resource", {})
        dataset_id = resource.get("defaultDatasetId")
        
        if not dataset_id:
            return JSONResponse(content={"message": "No dataset ID found"}, status_code=400)
            
        # Fetch dataset items
        # Note: In production, you might want to use the Apify Client or handle pagination
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        response = requests.get(dataset_url)
        
        if response.status_code != 200:
            logging.error(f"Failed to fetch dataset: {response.text}")
            return JSONResponse(content={"message": "Failed to fetch dataset"}, status_code=500)
            
        items = response.json()
        processed_count = 0
        
        for item in items:
            # Apify Facebook scrapers usually return 'text', 'message', or 'post_text'
            text = item.get("text") or item.get("message") or item.get("post_text") or ""
            
            # Extract username/author
            # Different scrapers use different fields. Common ones:
            user_data = item.get("user") or item.get("author") or {}
            username = user_data.get("name") or user_data.get("username") or item.get("userName") or "Facebook User"
            
            logging.debug(f"Extracting from item: {str(item)[:100]}...")
            logging.debug(f"Extracted username: {username}")
            
            if text:
                # Run analysis
                result = await predict_text(text)
                
                if not result.get("error"):
                    # Log to DB
                    log_analysis(result["text"], result["category"], result["confidence"], result["language"], username, "Facebook")
                    processed_count += 1
                    
        return JSONResponse(content={"message": "Processed successfully", "count": processed_count})
        
    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)

# --- Database Setup ---
import sqlite3
from datetime import datetime

DB_NAME = "hatespeech.db"

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
                  username TEXT,
                  platform TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Attempt to add columns if they don't exist (migration)
    try:
        c.execute("ALTER TABLE logs ADD COLUMN username TEXT")
    except sqlite3.OperationalError:
        pass # Column likely exists
        
    try:
        c.execute("ALTER TABLE logs ADD COLUMN platform TEXT")
    except sqlite3.OperationalError:
        pass # Column likely exists

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

import agents.db
agents.db.init_agent_tables()

import traceback

def log_analysis(text, category, confidence, language, username="Anonymous", platform="Web"):
    logging.debug(f"Attempting to insert log: {text[:20]}..., {category}, {username}, {platform}")
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO logs (text, category, confidence, language, username, platform) VALUES (?, ?, ?, ?, ?, ?)",
                  (text, category, confidence, language, username, platform))
        conn.commit()
        conn.close()
        logging.debug("Insert successful")
    except Exception as e:
        logging.error(f"DB Error: {e}")
        logging.error(traceback.format_exc())

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
    import random
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
        log_analysis(text, cat, conf, "English", username, "Facebook")

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
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
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
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return JSONResponse(content={"error": False, "data": rows})
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)

@app.get("/live-feed")
async def get_live_feed():
    """Get recent analyzed content (all categories)"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 50")
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return JSONResponse(content={"error": False, "data": rows})
    except Exception as e:
        return JSONResponse(content={"error": True, "message": str(e)}, status_code=500)

# --- Multi-agent orchestration layer (additive; see agents/ package) ---
from agents.router import router as agents_router
app.include_router(agents_router)


