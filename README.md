📺 VidChat AI - YouTube RAG Explainer

VidChat AI is a sleek, dark-mode enabled web application that allows users to chat with any YouTube video.

By simply pasting a YouTube link, the app processes the video audio, transcribes it, and uses advanced AI (Llama 3 on Groq) to answer your questions based on the video's content.

✨ Key Features

No System Dependencies: Does NOT require FFmpeg to be installed on your machine.

Blazing Fast: Uses Groq's LPU for near-instant transcription and text generation.

Modern UI: Beautiful Streamlit interface with dark mode and responsive design.

RAG Architecture: Uses Retrieval-Augmented Generation to give accurate answers based only on the video.

Cloud Transcription: Offloads heavy audio processing to the cloud.

🛠️ Tech Stack

Frontend: Streamlit (Python web framework)

LLM & Transcription: Groq API (Llama 3.3 & Whisper Large V3)

Vector Database: ChromaDB (for storing text chunks)

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

Video Tools: yt-dlp (for downloading audio)

🚀 Installation & Setup

1. Prerequisites

Python 3.8 or higher installed.

A Groq API Key (Get one for free at console.groq.com).

2. Clone or Download

Download the project files to a folder on your computer.

3. Install Libraries

Open your terminal/command prompt in the project folder and run:

pip install -r requirements.txt


4. Add API Key

Open app.py in a text editor and look for line 55:

GROQ_API_KEY = "paste_your_api_key_here"


Replace "paste_your_api_key_here" with your actual key, e.g., "gsk_8A...".

5. Run the App

In your terminal, run:

streamlit run app.py


A browser window will open automatically (usually at http://localhost:8501).

📂 Project Structure

app.py: The main application code containing UI and Logic.

requirements.txt: List of Python libraries required.

README.md: This documentation file.

⚠️ Note on Large Videos

The Groq Audio API has a file size limit (typically 25MB). This app works best with:

Standard YouTube videos (10-30 mins).

Podcasts or Talks (up to 45-60 mins depending on audio quality).

If a video is too long, the API might reject the file.

🤝 Troubleshooting

"Download failed: No file found"

This usually happens if the video is region-locked or premium content. Try a different public video.

"Error code: 400 - model decommissioned"

Groq updates models frequently. If distil-whisper fails, the code is already set to use whisper-large-v3 which is the current standard.
