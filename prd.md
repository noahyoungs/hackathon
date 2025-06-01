# Hackathon Project Spec Sheet: AI Medical Assistant

## Overview
An AI-powered medical tool that transcribes patient-healthcare professional interactions and extracts key insights. The system updates a simulated medical record in real time and helps doctors view relevant patient history dynamically based on conversation context.

---

## Components

### 1. Patient Scenario Setup
- One mock patient with a pre-defined health background.
- One defined scenario describing the patient’s current concern or reason for visit.

### 2. Fake Medical Record (Initial)
- Contains basic demographics and health history:
  - Name, Age, Sex
  - Medical conditions
  - Medications
  - Allergies
  - Past visits/notes

### 3. Nurse Interaction
- Simulated conversation between patient and nurse.
- Transcription using a speech-to-text engine (e.g., Whisper).
- AI-generated summary of key info:
  - Vital signs
  - Symptoms reported
  - Lifestyle notes
  - Other relevant history
- Updates appended to the patient’s medical record.

### 4. Doctor Pre-Visit Summary
- Doctor sees:
  - Full updated medical record
  - Patient’s current scenario summary
  - Key notes from the nurse

### 5. Doctor Interaction with Real-Time Insights
- Transcription of doctor-patient conversation in real time.
- Vector similarity search (e.g., using OpenAI embeddings + Pinecone/FAISS):
  - Match segments of conversation to most relevant parts of the medical record.
  - Highlight relevant records in sync with discussion.
- Display updated summary and linked references.

## 🛠️ Tech Stack

### Backend
- **Framework:** Python + FastAPI
- **WebSocket:** FastAPI native WebSocket (`from fastapi import WebSocket`)
- **Transcription:** OpenAI Whisper
- **AI Tasks:** GPT-4 (summarization), vector similarity (e.g., cosine similarity with sentence-transformers)
- **Storage:** In-memory Python objects (dicts) — no DB needed

### Frontend
- **Served From:** FastAPI `StaticFiles`
- **Tech:** Vanilla HTML/CSS/JS
- **WebSocket Client:** Native browser WebSocket API
- **Display:** 
  - Doctor screen with:  
    - Full medical record summary  
    - Live transcription  
    - Live-updating “Relevant Record Snippets”
