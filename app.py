from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import sqlite3
import json
from datetime import datetime
import os
from typing import Optional
import base64
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgriAR Backend", description="Smart farming assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect('agri_data.db')
    cursor = conn.cursor()
    
    # Crops table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            soil_type TEXT NOT NULL,
            season TEXT NOT NULL,
            water_requirement TEXT,
            fertilizer_requirement TEXT,
            estimated_yield TEXT,
            sustainable_practices TEXT
        )
    ''')
    
    # Diseases table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            crop_type TEXT NOT NULL,
            symptoms TEXT,
            treatment TEXT,
            eco_friendly_treatment TEXT,
            severity_level TEXT
        )
    ''')
    
    # Pesticides table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pesticides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            disease_target TEXT NOT NULL,
            application_method TEXT,
            eco_friendly BOOLEAN,
            safety_level TEXT
        )
    ''')
    
    # User analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_location TEXT,
            analysis_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result TEXT
        )
    ''')
    
    # Insert sample data
    sample_crops = [
        ("Rice", "clay", "monsoon", "high", "nitrogen-rich", "4-6 tons/hectare", "Use organic fertilizers, crop rotation"),
        ("Wheat", "loamy", "winter", "medium", "phosphorus-rich", "3-4 tons/hectare", "Drip irrigation, green manure"),
        ("Tomato", "sandy", "summer", "high", "potassium-rich", "40-50 tons/hectare", "Mulching, companion planting"),
        ("Cotton", "loamy", "summer", "medium", "balanced NPK", "2-3 tons/hectare", "Integrated pest management"),
        ("Sugarcane", "clay", "monsoon", "very high", "nitrogen-heavy", "80-100 tons/hectare", "Precision agriculture"),
        ("Maize", "sandy", "monsoon", "medium", "nitrogen-rich", "5-7 tons/hectare", "Contour farming, cover crops")
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO crops (name, soil_type, season, water_requirement, fertilizer_requirement, estimated_yield, sustainable_practices)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', sample_crops)
    
    sample_diseases = [
        ("Leaf Blight", "Rice", "Brown spots on leaves", "Copper fungicide spray", "Neem oil treatment", "moderate"),
        ("Powdery Mildew", "Wheat", "White powdery coating", "Sulfur-based fungicide", "Baking soda spray", "mild"),
        ("Bacterial Wilt", "Tomato", "Yellowing and wilting", "Copper bactericide", "Cinnamon bark extract", "severe"),
        ("Bollworm", "Cotton", "Holes in leaves and bolls", "Bt spray", "Neem oil and garlic spray", "moderate"),
        ("Red Rot", "Sugarcane", "Red discoloration", "Carbendazim", "Trichoderma treatment", "severe"),
        ("Corn Borer", "Maize", "Holes in stalks", "Insecticide spray", "Bacillus thuringiensis", "moderate")
    ]
    
    cursor.executemany('''
        INSERT OR IGNORE INTO diseases (name, crop_type, symptoms, treatment, eco_friendly_treatment, severity_level)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_diseases)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# AI Models (Mock implementations - replace with actual trained models)
class SoilAnalyzer:
    def __init__(self):
        self.soil_types = ["loamy", "sandy", "clay"]
        
    def analyze_soil(self, image_array):
        # Mock soil analysis - replace with actual CNN model
        # This is a simplified texture analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Mock classification based on intensity
        if mean_intensity > 150:
            return "sandy"
        elif mean_intensity < 100:
            return "clay"
        else:
            return "loamy"
    
    def get_fertility_score(self, soil_type):
        fertility_map = {
            "loamy": 0.8,
            "sandy": 0.5,
            "clay": 0.7
        }
        return fertility_map.get(soil_type, 0.6)

class DiseaseDetector:
    def __init__(self):
        self.diseases = [
            "Leaf Blight", "Powdery Mildew", "Bacterial Wilt", 
            "Bollworm", "Red Rot", "Corn Borer", "Healthy"
        ]
        
    def detect_disease(self, image_array):
        # Mock disease detection - replace with actual CNN model
        # This is a simplified color-based detection
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Detect brown spots (leaf blight)
        brown_lower = np.array([10, 50, 50])
        brown_upper = np.array([30, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # Detect white patches (powdery mildew)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        brown_ratio = np.sum(brown_mask) / (image_array.shape[0] * image_array.shape[1] * 255)
        white_ratio = np.sum(white_mask) / (image_array.shape[0] * image_array.shape[1] * 255)
        
        if brown_ratio > 0.1:
            return "Leaf Blight", 0.8
        elif white_ratio > 0.1:
            return "Powdery Mildew", 0.7
        else:
            return "Healthy", 0.9

# Initialize AI models
soil_analyzer = SoilAnalyzer()
disease_detector = DiseaseDetector()

# Weather API (mock implementation)
def get_weather_data(lat, lng):
    # Mock weather data - replace with actual API
    return {
        "temperature": 25,
        "humidity": 70,
        "rainfall": 10,
        "season": "monsoon"
    }

# Database helpers
def get_crop_recommendations(soil_type, season):
    conn = sqlite3.connect('agri_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM crops 
        WHERE soil_type = ? AND season = ?
    ''', (soil_type, season))
    
    crops = cursor.fetchall()
    conn.close()
    
    return [
        {
            "name": crop[1],
            "soil_type": crop[2],
            "season": crop[3],
            "water_requirement": crop[4],
            "fertilizer_requirement": crop[5],
            "estimated_yield": crop[6],
            "sustainable_practices": crop[7]
        }
        for crop in crops
    ]

def get_disease_info(disease_name):
    conn = sqlite3.connect('agri_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM diseases 
        WHERE name = ?
    ''', (disease_name,))
    
    disease = cursor.fetchone()
    conn.close()
    
    if disease:
        return {
            "name": disease[1],
            "crop_type": disease[2],
            "symptoms": disease[3],
            "treatment": disease[4],
            "eco_friendly_treatment": disease[5],
            "severity_level": disease[6]
        }
    return None

def log_user_activity(location, analysis_type, result):
    conn = sqlite3.connect('agri_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO user_analytics (user_location, analysis_type, result)
        VALUES (?, ?, ?)
    ''', (location, analysis_type, json.dumps(result)))
    
    conn.commit()
    conn.close()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AgriAR Backend API is running!"}

@app.post("/analyze/soil")
async def analyze_soil(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)
        
        # Analyze soil
        soil_type = soil_analyzer.analyze_soil(image_array)
        fertility_score = soil_analyzer.get_fertility_score(soil_type)
        
        # Get weather data
        weather = get_weather_data(latitude, longitude)
        
        # Get crop recommendations
        recommendations = get_crop_recommendations(soil_type, weather["season"])
        
        result = {
            "soil_analysis": {
                "soil_type": soil_type,
                "fertility_score": fertility_score,
                "ph_level": "6.5-7.0",  # Mock data
                "organic_matter": "2.5%"  # Mock data
            },
            "weather": weather,
            "crop_recommendations": recommendations,
            "location": {
                "latitude": latitude,
                "longitude": longitude
            }
        }
        
        # Log activity
        log_user_activity(f"{latitude},{longitude}", "soil_analysis", result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in soil analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/disease")
async def analyze_disease(
    file: UploadFile = File(...),
    crop_type: str = Form(...)
):
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)
        
        # Detect disease
        disease_name, confidence = disease_detector.detect_disease(image_array)
        
        # Get disease information
        disease_info = get_disease_info(disease_name)
        
        result = {
            "disease_detection": {
                "disease_name": disease_name,
                "confidence": confidence,
                "crop_type": crop_type
            },
            "disease_info": disease_info,
            "treatment_recommendations": {
                "immediate_action": "Isolate affected plants",
                "organic_treatment": disease_info["eco_friendly_treatment"] if disease_info else "Neem oil spray",
                "chemical_treatment": disease_info["treatment"] if disease_info else "Consult agricultural expert"
            }
        }
        
        # Log activity
        log_user_activity("unknown", "disease_analysis", result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in disease analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/dashboard")
async def get_analytics():
    try:
        conn = sqlite3.connect('agri_data.db')
        cursor = conn.cursor()
        
        # Get analysis counts
        cursor.execute('''
            SELECT analysis_type, COUNT(*) as count
            FROM user_analytics
            GROUP BY analysis_type
        ''')
        analysis_counts = dict(cursor.fetchall())
        
        # Get recent activities
        cursor.execute('''
            SELECT analysis_type, timestamp, user_location
            FROM user_analytics
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        recent_activities = cursor.fetchall()
        
        conn.close()
        
        return {
            "analysis_counts": analysis_counts,
            "recent_activities": [
                {
                    "type": activity[0],
                    "timestamp": activity[1],
                    "location": activity[2]
                }
                for activity in recent_activities
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/community/disease-map")
async def get_disease_map():
    try:
        conn = sqlite3.connect('agri_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_location, result, timestamp
            FROM user_analytics
            WHERE analysis_type = 'disease_analysis'
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        disease_reports = cursor.fetchall()
        conn.close()
        
        disease_map = []
        for report in disease_reports:
            try:
                result = json.loads(report[1])
                if result["disease_detection"]["disease_name"] != "Healthy":
                    location = report[0].split(",") if report[0] != "unknown" else ["0", "0"]
                    disease_map.append({
                        "location": {
                            "latitude": float(location[0]),
                            "longitude": float(location[1])
                        },
                        "disease": result["disease_detection"]["disease_name"],
                        "crop": result["disease_detection"]["crop_type"],
                        "timestamp": report[2]
                    })
            except:
                continue
        
        return {"disease_outbreaks": disease_map}
        
    except Exception as e:
        logger.error(f"Error in disease map: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/process")
async def process_voice(
    audio_data: str = Form(...),
    language: str = Form(default="en")
):
    try:
        # Mock voice processing - replace with actual speech recognition
        # This would typically use Google Speech-to-Text or similar
        
        # Simulate processing
        mock_responses = {
            "en": "Based on your query, I recommend checking your soil pH and using organic fertilizers.",
            "hi": "आपके प्रश्न के आधार पर, मैं मिट्टी की pH जांचने और जैविक उर्वरकों का उपयोग करने की सलाह देता हूं।",
            "te": "మీ ప్రశ్న ఆధారంగా, మట్టి pH తనిఖీ చేయాలని మరియు సేంద్రీయ ఎరువులను ఉపయోగించాలని నేను సిఫార్సు చేస్తున్నాను।"
        }
        
        response = mock_responses.get(language, mock_responses["en"])
        
        return {
            "response": response,
            "language": language,
            "confidence": 0.9
        }
        
    except Exception as e:
        logger.error(f"Error in voice processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(message: dict):
    try:
        # Mock WhatsApp integration
        # This would typically integrate with WhatsApp Business API
        
        user_message = message.get("message", "")
        user_phone = message.get("phone", "")
        
        # Simple keyword-based responses
        if "soil" in user_message.lower():
            response = "To analyze soil, please visit our website and upload a soil image."
        elif "disease" in user_message.lower():
            response = "For disease detection, take a clear photo of affected leaves and upload on our platform."
        else:
            response = "Hello! I'm your AgriAR assistant. I can help with soil analysis and disease detection."
        
        return {
            "response": response,
            "phone": user_phone
        }
        
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)