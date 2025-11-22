import os
import streamlit as st
import yt_dlp
import whisper
import torch
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import uuid
import tempfile
import shutil 

# --- CONFIGURATION ---
st.set_page_config(page_title="YouTube Video Chat", page_icon="📺", layout="wide")

# 👇👇👇 PASTE YOUR GROQ API KEY HERE 👇👇👇
GROQ_API_KEY = "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"
# 👆👆👆 PASTE YOUR GROQ API KEY HERE 👆👆👆

# --- SYSTEM CHECKS ---
def check_system_deps():
    """Checks if FFmpeg AND FFprobe are installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        st.error("⚠️ **Critical Dependency Missing: FFmpeg/FFprobe**")
        st.markdown(f"""
        This app requires both **FFmpeg** and **FFprobe** to process audio.
        
        - FFmpeg found: `{ffmpeg_path}`
        - FFprobe found: `{ffprobe_path}`
        
        **How to fix:**
        1. **Streamlit Cloud:** Ensure `packages.txt` exists in your repo and contains the word `ffmpeg`.
        2. **Local (Windows):** Download the 'full' or 'essentials' build of FFmpeg (not just the executable) and ensure both `ffmpeg.exe` and `ffprobe.exe` are in your PATH.
        """)
        st.stop()
    return ffmpeg_path

# Run check immediately and get path
FFMPEG_PATH = check_system_deps()

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

@st.cache_resource
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- HELPER FUNCTIONS ---

def download_audio(youtube_url):
    """Downloads audio to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_filename = temp_audio.name

    # We explicitly provide the ffmpeg location to yt-dlp to avoid path errors
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': FFMPEG_PATH, # EXPLICITLY SET FFMPEG LOCATION
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': temp_filename,
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return temp_filename
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """Manually splits text into chunks with overlap."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (chunk_size * 0.5):
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= end: start = end
            
    return chunks

def process_video(url):
    """Orchestrates the download, transcription, and indexing."""
    status = st.status("Processing video...", expanded=True)
    
    try:
        # 1. Download
        status.write("📥 Downloading audio...")
        audio_path = download_audio(url)
        
        # 2. Transcribe
        status.write("🎙️ Transcribing audio (this takes a moment)...")
        whisper_model = load_whisper_model()
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        
        # 3. Split
        status.write("✂️ Splitting text...")
        chunks = split_text(transcription)
        
        # 4. Embed & Store
        status.write("🧠 Indexing content...")
        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Setup ChromaDB
        chroma_client = chromadb.Client()
        collection_name = "video_rag_" + str(uuid.uuid4())
        collection = chroma_client.create_collection(name=collection_name)
        
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        # Cleanup
        try:
            os.remove(audio_path)
        except:
            pass
        
        status.update(label="✅ Video Processed!", state="complete", expanded=False)
        return collection

    except Exception as e:
        status.update(label="❌ Error", state="error", expanded=True)
        st.error(f"An error occurred: {str(e)}")
        return None

# --- MAIN APPLICATION UI ---

st.title("📺 YouTube Video RAG Explainer")
st.caption("Powered by Groq, Llama 3, & Streamlit")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_collection" not in st.session_state:
    st.session_state.vector_collection = None

# Sidebar for Inputs
with st.sidebar:
    st.header("Video Setup")
    youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/...")
    
    if st.button("Process Video", type="primary"):
        if not GROQ_API_KEY or GROQ_API_KEY == "paste_your_api_key_here":
            st.error("❌ Groq API Key is missing! Please paste it in the code.")
        elif not youtube_url:
            st.warning("Please enter a URL.")
        else:
            st.session_state.messages = []
            st.session_state.vector_collection = process_video(youtube_url)

# Chat Interface
if not st.session_state.vector_collection:
    st.info("👈 Please enter a YouTube URL in the sidebar and click 'Process Video' to start.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                embedding_model = load_embedding_model()
                query_embedding = embedding_model.encode([prompt]).tolist()
                
                results = st.session_state.vector_collection.query(
                    query_embeddings=query_embedding,
                    n_results=3
                )
                
                if results['documents']:
                    context_text = "\n\n".join(results['documents'][0])
                else:
                    context_text = "No relevant context found."
                
                client = Groq(api_key=GROQ_API_KEY)
                completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant. Answer based ONLY on the context provided."
                        },
                        {
                            "role": "user", 
                            "content": f"Context:\n{context_text}\n\nQuestion:\n{prompt}"
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                    max_tokens=1024,
                    stream=True
                )
                
                full_response = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error generating response: {e}")
