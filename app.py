import os
import streamlit as st
import yt_dlp
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import uuid
import tempfile
import shutil

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="VidChat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Headers */
    h1 {
        color: #FF4B4B !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #FAFAFA !important;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 👇👇👇 PASTE YOUR GROQ API KEY HERE 👇👇👇
GROQ_API_KEY = "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"
# 👆👆👆 PASTE YOUR GROQ API KEY HERE 👆👆👆

# --- CACHED RESOURCE LOADING ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPER FUNCTIONS ---
def download_audio_raw(youtube_url):
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'nocheckcertificate': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        files = os.listdir(temp_dir)
        if not files:
            raise Exception("No file found.")
        return os.path.join(temp_dir, files[0])
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise Exception(f"Download failed: {str(e)}")

def split_text(text, chunk_size=1000, chunk_overlap=100):
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
    status = st.status("🚀 Processing video...", expanded=True)
    audio_path = None
    try:
        client = Groq(api_key=api_key)

        status.write("📥 Downloading audio stream...")
        audio_path = download_audio_raw(url)
        
        if os.path.getsize(audio_path) == 0:
             raise Exception("Audio file is empty.")

        status.write("✨ Transcribing with Groq Whisper...")
        with open(audio_path, "rb") as file:
            transcription_obj = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
        transcription = transcription_obj.text
        
        status.write("🧠 Analyzing content structure...")
        chunks = split_text(transcription)
        
        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        chroma_client = chromadb.Client()
        collection_name = "video_rag_" + str(uuid.uuid4())
        collection = chroma_client.create_collection(name=collection_name)
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
        return collection

    except Exception as e:
        status.update(label="❌ Error Occurred", state="error", expanded=True)
        st.error(f"Error: {str(e)}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                shutil.rmtree(os.path.dirname(audio_path))
            except:
                pass

# --- MAIN APPLICATION UI ---

# Sidebar
with st.sidebar:
    st.title("🎬 VidChat AI")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    youtube_url = st.text_input("Paste YouTube Link", placeholder="https://youtube.com/...")
    
    st.markdown("### 📝 Instructions")
    st.info(
        """
        1. Enter your **Groq API Key** in the code.
        2. Paste a **YouTube URL**.
        3. Click **Analyze Video**.
        4. Chat with the AI!
        """
    )
    
    process_btn = st.button("✨ Analyze Video", type="primary")
    
    st.markdown("---")
    st.caption("Powered by Groq, Llama 3 & ChromaDB")

# Main Content Area
st.title("🤖 Chat with Any YouTube Video")
st.markdown("#### Ask questions, get summaries, and explore video content instantly.")

# Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! Paste a video link in the sidebar to get started."}
    ]

if "vector_collection" not in st.session_state:
    st.session_state.vector_collection = None

# Process Logic
if process_btn:
    if not GROQ_API_KEY or GROQ_API_KEY == "paste_your_api_key_here":
        st.sidebar.error("❌ Groq API Key missing in code!")
    elif not youtube_url:
        st.sidebar.warning("⚠️ Please enter a URL.")
    else:
        st.session_state.messages = [{"role": "assistant", "content": "✅ Video analyzed! Ask me anything about it."}]
        st.session_state.vector_collection = process_video(youtube_url, GROQ_API_KEY)

# Chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about the video..."):
    if not st.session_state.vector_collection:
        st.warning("⚠️ Please analyze a video first!")
    else:
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
                
                context = "\n\n".join(results['documents'][0]) if results['documents'] else "No context."
                
                client = Groq(api_key=GROQ_API_KEY)
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant. Answer based on the provided context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                    max_tokens=1024,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Generation Error: {e}")
