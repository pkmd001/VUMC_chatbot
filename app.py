import streamlit as st
import openai
import requests
import PyPDF2
import pdfplumber
from pathlib import Path
import hashlib
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional, Tuple
import base64
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="VUMC Trauma Guidelines Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile-friendly interface
st.markdown("""
<style>
    /* Mobile-friendly adjustments */
    .stApp {
        max-width: 100%;
    }
    
    /* Message styling */
    .user-message {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
        margin-left: 20%;
    }
    
    .bot-message {
        background-color: #E8E8E8;
        color: #333;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        margin-right: 20%;
    }
    
    .source-link {
        color: #007AFF;
        text-decoration: none;
        font-size: 0.9em;
    }
    
    /* Mobile button styling */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        padding: 10px;
    }
    
    /* Sidebar mobile friendly */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            margin-left: 0;
            margin-right: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False
if 'embeddings_index' not in st.session_state:
    st.session_state.embeddings_index = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'chunk_metadata' not in st.session_state:
    st.session_state.chunk_metadata = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Guidelines dictionary
GUIDELINES = {
    "Kidney Injury Grading": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/KIDNEY_INJURY_GRADES-CODES_031522.pdf",
    "Liver Injury Grading": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/LIVER_INJURY_GRADES-CODES_031522.pdf",
    "Pancreas Injury Grading": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/PANCREAS_INJURY_GRADES-CODES_051920_MOD.pdf",
    "Spleen Injury Grading": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/SPLEEN_INJURY_GRADES-CODES_031522.pdf",
    "Traumatic Brain Injury": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/TBI-Pathways-PMG-v2023.pdf",
    "Massive Transfusion": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/VUMC%20Massive%20Transfusion%20Protocol.pdf",
    "Geriatric Trauma": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/Geriatric%20Trauma%20PMG_2024_final.pdf",
    "C-Collar Clearance": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Protocols/C-Collar%20Clearance%20PMG%202023.pdf",
    "Rib Fracture": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/Rib%20Fx%20PMG%20v2024.pdf",
    "ECMO": "https://www.vumc.org/trauma-and-scc/sites/default/files/public_files/ECMO_PMG_1.2025.pdf"
}

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def download_pdf(url: str, name: str) -> Optional[bytes]:
    """Download PDF and return bytes."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading {name}: {e}")
        return None

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    text = ""
    
    # Try pdfplumber first
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"PyPDF2 also failed: {e}")
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def load_pdfs_and_create_index():
    """Load all PDFs and create the search index."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    embedder = load_embedding_model()
    text_chunks = []
    chunk_metadata = []
    
    for idx, (name, url) in enumerate(GUIDELINES.items()):
        progress = (idx + 1) / len(GUIDELINES)
        progress_bar.progress(progress)
        status_text.text(f"Loading {name}...")
        
        # Download PDF
        pdf_bytes = download_pdf(url, name)
        if pdf_bytes:
            # Extract text
            text = extract_pdf_text(pdf_bytes)
            if text:
                # Create chunks
                chunks = chunk_text(text)
                for chunk in chunks:
                    text_chunks.append(chunk)
                    chunk_metadata.append({
                        'source': name,
                        'url': url
                    })
    
    status_text.text("Creating embeddings index...")
    
    if text_chunks:
        # Generate embeddings
        embeddings = embedder.encode(text_chunks, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Store in session state
        st.session_state.embeddings_index = index
        st.session_state.text_chunks = text_chunks
        st.session_state.chunk_metadata = chunk_metadata
        st.session_state.pdf_loaded = True
        
        status_text.text(f"✅ Loaded {len(GUIDELINES)} PDFs with {len(text_chunks)} searchable chunks!")
    else:
        status_text.text("❌ No PDFs could be loaded")
    
    progress_bar.empty()

def semantic_search(query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
    """Search for relevant chunks using semantic similarity."""
    if not st.session_state.embeddings_index:
        return []
    
    embedder = load_embedding_model()
    
    # Encode query
    query_embedding = embedder.encode([query])
    
    # Search
    distances, indices = st.session_state.embeddings_index.search(
        query_embedding.astype('float32'), top_k
    )
    
    results = []
    for idx in indices[0]:
        if idx < len(st.session_state.text_chunks):
            results.append((
                st.session_state.text_chunks[idx], 
                st.session_state.chunk_metadata[idx]
            ))
    
    return results

def chat_with_context(query: str, api_key: str) -> str:
    """Process query using PDF content context."""
    # Search for relevant content
    relevant_chunks = semantic_search(query, top_k=3)
    
    # Build context from relevant chunks
    context = ""
    sources = set()
    
    for chunk, metadata in relevant_chunks:
        context += f"\n{chunk}\n"
        sources.add(metadata['source'])
    
    # Create prompt with context
    system_prompt = """You are a medical assistant specialized in VUMC Trauma and Surgical Critical Care guidelines. 
    Use the provided context from the guidelines to answer questions accurately. 
    Always cite which guideline the information comes from.
    If the context doesn't contain relevant information, say so clearly.
    Keep answers concise and mobile-friendly."""
    
    user_prompt = f"""Context from VUMC guidelines:
{context}

Question: {query}

Please provide a clear, concise answer based on the context above. Cite the specific guideline(s) used."""

    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # Add source information
        if sources:
            answer += f"\n\n📚 **Sources:** {', '.join(sources)}"
        
        return answer, sources
        
    except Exception as e:
        return f"Error: {str(e)}", []

# Header
st.title("🏥 VUMC Trauma Guidelines Assistant")
st.caption("Ask questions about VUMC trauma and surgical critical care guidelines")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        help="Enter your OpenAI API key to enable the chatbot"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        st.success("✅ API Key set!")
    else:
        st.warning("⚠️ Please enter your API key")
    
    st.divider()
    
    # PDF Loading
    st.header("📄 PDF Guidelines")
    
    if not st.session_state.pdf_loaded:
        if st.button("🔄 Load All PDFs", type="primary", disabled=not api_key):
            with st.spinner("Loading PDFs... This may take a few minutes"):
                load_pdfs_and_create_index()
    else:
        st.success(f"✅ {len(GUIDELINES)} PDFs loaded")
        st.info(f"📊 {len(st.session_state.text_chunks)} searchable chunks")
        
        if st.button("🔄 Reload PDFs"):
            st.session_state.pdf_loaded = False
            st.session_state.embeddings_index = None
            st.session_state.text_chunks = []
            st.session_state.chunk_metadata = []
            st.experimental_rerun()
    
    st.divider()
    
    # Quick Questions
    st.header("💡 Quick Questions")
    quick_questions = [
        "What are the grades for spleen injury?",
        "When to activate massive transfusion?",
        "TBI management pathway?",
        "C-collar clearance criteria?",
        "ECMO indications?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{question}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.experimental_rerun()
    
    st.divider()
    
    # Available Guidelines
    st.header("📚 Available Guidelines")
    for name in GUIDELINES.keys():
        st.text(f"• {name}")

# Main chat interface
main_container = st.container()

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message["content"]}</div>', 
                   unsafe_allow_html=True)

# Chat input
if api_key and st.session_state.pdf_loaded:
    user_input = st.chat_input("Ask about trauma guidelines...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process with loading spinner
        with st.spinner("Thinking..."):
            response, sources = chat_with_context(user_input, api_key)
        
        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.experimental_rerun()

elif not api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to begin")
elif not st.session_state.pdf_loaded:
    st.info("👈 Please load the PDF guidelines in the sidebar to begin")

# Footer with instructions
with st.expander("📱 Mobile Usage Tips"):
    st.markdown("""
    **For best mobile experience:**
    1. Add this page to your home screen for app-like access
    2. Use landscape mode for better readability
    3. Tap the sidebar arrow to show/hide settings
    4. Voice input: Use your keyboard's voice-to-text feature
    
    **How to use:**
    1. Enter your OpenAI API key (saved locally)
    2. Click "Load All PDFs" (one-time setup)
    3. Ask questions in the chat or use quick questions
    4. Get answers based on actual VUMC guidelines!
    """)

# Requirements.txt content
st.sidebar.divider()
if st.sidebar.button("📋 Show requirements.txt"):
    st.sidebar.code("""
streamlit
openai
PyPDF2
pdfplumber
sentence-transformers
faiss-cpu
numpy
requests
    """, language="text")