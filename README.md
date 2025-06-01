# 🏥 AI Medical Assistant

A real-time AI-powered medical tool that transcribes patient-healthcare professional interactions and extracts key insights using OpenAI Whisper and GPT-4.

## Features

### ✅ Phase 1: Nurse Interaction Interface (Completed)
- 🎤 **Real-time Voice Recording** - Click-to-record microphone interface
- 📝 **Live Transcription** - Powered by OpenAI Whisper
- 🤖 **AI Summarization** - GPT-4 generates medical summaries
- 💾 **Data Persistence** - Saves summaries to JSON files
- 🎨 **Beautiful UI** - Modern, responsive interface

### 🚧 Coming Next
- Doctor-patient interaction with real-time insights
- Vector similarity search for relevant medical history
- WebSocket real-time updates
- Medical record integration and updates

## Tech Stack

- **Backend**: FastAPI + Python
- **AI**: OpenAI Whisper (transcription) + GPT-4 (summarization)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Storage**: JSON files (simple, no database needed)

## Setup Instructions

### 1. Prerequisites
- Python 3.8+ (tested with Python 3.13)
- OpenAI API account and key
- Microphone access in your browser

### 2. Installation

```bash
# Clone or download the project
cd hackathon

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi "uvicorn[standard]" openai python-multipart aiofiles python-dotenv websockets
pip install git+https://github.com/openai/whisper.git
```

### 3. Configuration

Create a `.env` file in the project root:

```bash
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Running the Application

```bash
# Start the server
python main.py

# Or alternatively:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at: http://localhost:8000

## Usage

### Nurse Interface Workflow

1. **Access the Interface**: Open http://localhost:8000 in your browser
2. **Patient Context**: See the current patient info (Sarah Michelle Johnson)
3. **Record Conversation**: 
   - Click the microphone button to start recording
   - Speak your nurse-patient conversation
   - Click again to stop recording
4. **View Transcription**: See the live transcription appear
5. **Generate Summary**: Click "Generate Summary" for AI analysis
6. **Save to Record**: Click "Save to Record" to persist the data

### Features in Action

- **Real-time Transcription**: Using Whisper's "base" model for speed
- **Medical Context**: GPT-4 receives patient medical history for context
- **Professional Summaries**: AI generates clinical documentation
- **Persistent Storage**: All interactions saved to `summaries/` directory

## File Structure

```
hackathon/
├── main.py                 # FastAPI application
├── medical_record.md       # Patient medical history
├── requirements.txt        # Python dependencies
├── summaries/             # Generated summaries directory
│   ├── all_summaries.json # Master file with all summaries
│   └── summary_*.json     # Individual session files
└── README.md              # This file
```

## API Endpoints

- `GET /` - Main nurse interface
- `POST /transcribe` - Upload audio for transcription
- `POST /summarize` - Generate medical summary
- `POST /save-summary` - Save summary to JSON
- `GET /summaries` - Retrieve all saved summaries

## Sample Medical Summary

The AI generates professional medical notes like:

```
**Chief Complaint**: Patient reports chest discomfort and shortness of breath with exertion over the past week.

**Vital Signs**: Blood pressure elevated at 145/92 mmHg, heart rate 88 bpm.

**Assessment**: Given patient's history of diabetes, hypertension, and family cardiac history, chest symptoms warrant further cardiac evaluation.

**Plan**: ECG ordered, cardiology referral initiated, continue current medications.
```

## Troubleshooting

### Common Issues

1. **Microphone not working**: Ensure browser permissions for microphone access
2. **OpenAI API errors**: Check your API key in `.env` file
3. **Whisper loading slowly**: First model load takes time, subsequent uses are faster
4. **Module not found**: Ensure virtual environment is activated and dependencies installed

### Performance Tips

- Use Chrome/Firefox for best microphone support
- Speak clearly for better transcription accuracy
- Keep recordings under 30 seconds for optimal processing
- The "base" Whisper model balances speed and accuracy

## Contributing

This is a hackathon project! Feel free to:
- Add new features
- Improve the UI/UX
- Optimize performance
- Add error handling
- Implement additional medical workflows

## Next Steps

Based on your PRD, the next features to implement:
1. Doctor interface with real-time insights
2. Vector similarity search using sentence-transformers
3. Medical record highlighting based on conversation
4. WebSocket real-time updates

## License

MIT License - Built for the hackathon! 