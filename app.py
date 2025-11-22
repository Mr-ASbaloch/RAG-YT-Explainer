import os
import streamlit as st
import yt_dlp
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import uuid
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="YouTube Video Chat", page_icon="📺", layout="wide")

# 👇👇👇 PASTE YOUR GROQ API KEY HERE 👇👇👇
GROQ_API_KEY = "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"
# 👆👆👆 PASTE YOUR GROQ API KEY HERE 👆👆👆

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_embedding_model():
    # We still use local embeddings as they don't require external tools
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPER FUNCTIONS ---

def download_audio_raw(youtube_url):
    """
    Downloads raw m4a audio without conversion to avoid FFmpeg dependency.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
        temp_filename = temp_audio.name

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/best', # Force m4a to avoid conversion
        'outtmpl': temp_filename,
        'quiet': True,
        'nocheckcertificate': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return temp_filename
    except Exception as e:
        # If m4a fails, try a generic best audio download
        try:
            ydl_opts['format'] = 'bestaudio/best'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            return temp_filename
        except Exception as final_e:
            raise Exception(f"Download failed: {str(final_e)}")

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

def process_video(url, api_key):
    """Orchestrates download, API transcription, and indexing."""
    status = st.status("Processing video...", expanded=True)
    
    try:
        client = Groq(api_key=api_key)

        # 1. Download Raw Audio
        status.write("📥 Downloading raw audio (No FFmpeg)...")
        audio_path = download_audio_raw(url)
        
        # 2. API Transcription (Groq Whisper)
        status.write("☁️ Sending to Groq for transcription...")
        
        # Open the file and send to Groq API
        with open(audio_path, "rb") as file:
            transcription_obj = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="distil-whisper-large-v3-en", # Groq's fast model
                response_format="json",
                language="en",
                temperature=0.0
            )
        transcription = transcription_obj.text
        
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
        st.markdown("If the file is too large (>25MB), Groq API might reject it.")
        return None

# --- MAIN APPLICATION UI ---

st.title("📺 YouTube Video RAG (No-Install Version)")
st.caption("Powered by Groq API (Audio & Chat) - No FFmpeg required")

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
            st.session_state.vector_collection = process_video(youtube_url, GROQ_API_KEY)

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
                # Retrieve Context
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
