from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import whisper
import openai
import json
import os
from datetime import datetime
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Medical Assistant", description="Real-time transcription and medical summarization")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables for medical data
medical_summaries = []
current_session = None
current_ai_summary = None  # Track current AI summary

# Load Whisper model (using smaller model for faster processing)
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model("tiny.en")
logger.info("Whisper model loaded successfully!")

# OpenAI client setup
logger.info("Setting up OpenAI client...")
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://gateway.ai.cloudflare.com/v1/c1e15f924beb8068d5ab8cf461f69ff8/hackathon/openai"
)
logger.info("OpenAI client setup complete")

# Load sentence transformer model and medical record vectors
logger.info("Loading sentence transformer model and medical record vectors...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load medical record vectors
try:
    medical_vectors = np.load('medical_record_vectors.npy')
    with open('medical_record.json', 'r') as f:
        medical_texts = json.load(f)
    logger.info(f"Loaded {len(medical_vectors)} medical record vectors")
except Exception as e:
    logger.error(f"Failed to load medical record vectors: {e}")
    medical_vectors = None
    medical_texts = None

# Load medical record
def load_medical_record():
    try:
        with open("medical_record.md", "r") as f:
            content = f.read()
            logger.info("Medical record loaded successfully")
            return content
    except FileNotFoundError:
        logger.warning("Medical record file not found")
        return "No medical record found"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main nurse interface"""
    logger.info("Serving nurse interface")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doctor-review", response_class=HTMLResponse)
async def doctor_review(request: Request):
    """Serve the doctor review interface"""
    logger.info("Serving doctor review interface")
    return templates.TemplateResponse("doctor_review.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe uploaded audio using Whisper"""
    try:
        logger.info(f"Starting transcription for audio file: {audio.filename}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Audio file saved to temporary path: {temp_file_path}")
        
        try:
            # Transcribe with Whisper
            logger.info("Starting Whisper transcription...")
            result = whisper_model.transcribe(temp_file_path)
            transcription = result["text"].strip()
            
            logger.info(f"Transcription completed successfully. Length: {len(transcription)} characters")
            logger.debug(f"Transcription content: {transcription[:100]}...")
            
            return {
                "success": True,
                "transcription": transcription
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            logger.info("Temporary audio file cleaned up")
            
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/summarize")
async def summarize_transcription(request: dict):
    """Generate medical summary using OpenAI GPT-4"""
    try:
        logger.info("Starting summarization request")
        logger.debug(f"Request data: {request}")
        
        transcription = request.get("transcription", "")
        if not transcription.strip():
            logger.warning("No transcription provided in summarization request")
            return {
                "success": False,
                "error": "No transcription provided"
            }
        
        logger.info(f"Transcription length: {len(transcription)} characters")
        logger.debug(f"Transcription content: {transcription[:200]}...")
        
        # Load medical record for context
        medical_record = load_medical_record()
        
        # Create prompt for medical summarization
        prompt = f"""You are a medical AI assistant helping a nurse document a patient interaction. 

Nurse-Patient Conversation Transcription:
{transcription}

Please provide a concise medical summary focusing on what was discussed.

Format the summary as a professional medical note suitable for a patient's record.

The doctors name is Dr. John Doe.
The time is Sundday June 1, 3:49PM 

Do not include placeholder stuff
"""

        logger.info("Sending request to OpenAI API...")
        logger.debug(f"OpenAI API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical documentation assistant. Provide clear, professional medical summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        logger.info("OpenAI API request completed successfully")
        
        summary = response.choices[0].message.content.strip()
        logger.info(f"Summary generated successfully. Length: {len(summary)} characters")
        logger.debug(f"Summary content: {summary[:200]}...")
        
        return {
            "success": True,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/save-summary")
async def save_summary(request: dict):
    """Save the summary to JSON file"""
    try:
        logger.info("Starting save summary request")
        logger.debug(f"Save request data: {request}")
        
        summary_data = {
            "timestamp": request.get("timestamp", datetime.now().isoformat()),
            "transcription": request.get("transcription", ""),
            "summary": request.get("summary", ""),
            "session_type": "nurse_interaction",
            "patient_id": "MR-2024-0347"
        }
        
        # Create summaries directory if it doesn't exist
        Path("summaries").mkdir(exist_ok=True)
        logger.info("Summaries directory ensured")
        
        # Save to JSON file
        filename = f"summaries/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary saved to individual file: {filename}")
        
        # Also append to a master summaries file
        master_file = "summaries/all_summaries.json"
        try:
            with open(master_file, "r") as f:
                all_summaries = json.load(f)
        except FileNotFoundError:
            logger.info("Master summaries file not found, creating new one")
            all_summaries = []
        
        all_summaries.append(summary_data)
        
        with open(master_file, "w") as f:
            json.dump(all_summaries, f, indent=2)
        
        logger.info(f"Summary added to master file: {master_file}")
        
        return {
            "success": True,
            "message": f"Summary saved to {filename}",
            "redirect_url": "/doctor-review"
        }
        
    except Exception as e:
        logger.error(f"Save summary failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/summaries")
async def get_summaries():
    """Get all saved summaries"""
    try:
        logger.info("Fetching all summaries")
        with open("summaries/all_summaries.json", "r") as f:
            summaries = json.load(f)
        logger.info(f"Successfully loaded {len(summaries)} summaries")
        return {
            "success": True,
            "summaries": summaries
        }
    except FileNotFoundError:
        logger.warning("Summaries file not found, returning empty list")
        return {
            "success": True,
            "summaries": []
        }
    except Exception as e:
        logger.error(f"Failed to fetch summaries: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/similarity_search")
async def similarity_search(request: dict):
    """Perform similarity search against medical record using query text"""
    try:
        logger.info("Starting similarity search request")
        
        query = request.get("query", "")
        if not query.strip():
            logger.warning("No query provided in similarity search request")
            return {
                "success": False,
                "error": "No query provided"
            }
        
        if medical_vectors is None or medical_texts is None:
            logger.error("Medical record vectors not loaded")
            return {
                "success": False,
                "error": "Medical record vectors not available"
            }
        
        logger.info(f"Query length: {len(query)} characters")
        logger.debug(f"Query content: {query}")
        
        # Generate embedding for the query
        query_embedding = sentence_model.encode([query], normalize_embeddings=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, medical_vectors)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        
        results = []
        for i in top_indices[:5]:  # Top 5 results
            similarity_score = float(similarities[i])
            if similarity_score > 0.1:  # Only include results with meaningful similarity
                results.append({
                    "text": medical_texts[i],
                    "similarity": similarity_score,
                    "index": int(i)
                })
        
        logger.info(f"Found {len(results)} relevant results")
        
        return {
            "success": True,
            "results": results,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Similarity search failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/get-ai-summary")
async def get_ai_summary():
    """Get the current AI summary for display in doctor review"""
    try:
        logger.info("Getting current AI summary")
        
        if current_ai_summary is None:
            logger.info("No AI summary available")
            return {
                "success": False,
                "error": "No AI summary available",
                "summary": None
            }
        
        logger.info("AI summary retrieved successfully")
        return {
            "success": True,
            "summary": current_ai_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI summary: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/set-ai-summary")
async def set_ai_summary(request: dict):
    """Set the AI summary from external source"""
    global current_ai_summary
    try:
        logger.info("Setting AI summary from external source")
        
        summary = request.get("summary", "")
        if not summary.strip():
            logger.warning("No summary provided in set AI summary request")
            return {
                "success": False,
                "error": "No summary provided"
            }
        
        current_ai_summary = summary
        logger.info(f"AI summary set successfully. Length: {len(summary)} characters")
        
        return {
            "success": True,
            "message": "AI summary set successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to set AI summary: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/check-summary-available")
async def check_summary_available():
    """Check if AI summary is available for conditional button rendering"""
    try:
        logger.info("Checking if AI summary is available")
        
        available = current_ai_summary is not None and current_ai_summary.strip() != ""
        
        return {
            "success": True,
            "available": available
        }
        
    except Exception as e:
        logger.error(f"Failed to check summary availability: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "available": False
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on localhost:12345")
    uvicorn.run(app, host="localhost", port=12345) 