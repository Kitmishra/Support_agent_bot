from fastapi import FastAPI, Query
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

app = FastAPI()

# Documentation sources
DOC_SOURCES = {
    "Segment": "https://segment.com/docs/?ref=nav",
    "mParticle": "https://docs.mparticle.com/",
    "Lytics": "https://docs.lytics.com/",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Load summarization model
SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn")

# FAISS index setup
EMBEDDING_DIM = 384  # MiniLM output size
INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
DOC_CHUNKS = []  # Stores document chunks for reference

def get_selenium_driver():
    """Sets up Selenium WebDriver with auto-update."""
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (no UI)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def fetch_documentation(url, use_selenium=False):
    """Fetch documentation content using Requests or Selenium."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        
        if use_selenium:
            driver = get_selenium_driver()
            driver.get(url)
            content = driver.page_source
            driver.quit()
        else:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            content = response.text
        
        soup = BeautifulSoup(content, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs) if paragraphs else "No content available."
    
    except Exception as e:
        print(f"❌ Error fetching {url}: {str(e)}")
        return "Error fetching this documentation."

def chunk_text(text, chunk_size=500):
    """Break text into smaller chunks to improve retrieval."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def load_and_index_documents():
    """Fetch and embed documentation sources into FAISS."""
    global INDEX, DOC_CHUNKS
    all_chunks = []

    for name, url in DOC_SOURCES.items():
        use_selenium = name == "Segment"  # Only use Selenium for Segment
        text = fetch_documentation(url, use_selenium)
        if text:
            chunks = chunk_text(text)
            if chunks:
                DOC_CHUNKS.extend(chunks)
                all_chunks.extend(chunks)
                print(f"🔹 {name.upper()} Documentation Indexed: {len(chunks)} chunks.")
    
    if all_chunks:
        embeddings = EMBEDDING_MODEL.encode(all_chunks, convert_to_numpy=True)
        INDEX.add(embeddings)
    else:
        print("⚠️ No documentation data loaded! FAISS index remains empty.")

def get_best_match(query_embedding):
    """Retrieve the best matching documentation snippet."""
    if INDEX.ntotal == 0:
        return "No documentation available."
    
    D, I = INDEX.search(np.array(query_embedding, dtype=np.float32), k=3)
    
    if I is None or len(I) == 0 or len(I[0]) == 0 or I[0][0] == -1:
        return "No relevant documentation found."
    
    best_matches = [DOC_CHUNKS[i] for i in I[0] if 0 <= i < len(DOC_CHUNKS)]
    return best_matches[0] if best_matches else "No relevant documentation found."

@app.get("/query/")
def query_chatbot(q: str = Query(..., description="User question")):
    """Handle user query by searching documentation."""
    print(f"🔍 User Query: {q}")
    query_embedding = EMBEDDING_MODEL.encode([q], convert_to_numpy=True)
    best_match_text = get_best_match(query_embedding)

    if "No relevant documentation found" in best_match_text:
        return {"query": q, "response": best_match_text}

    try:
        summary = SUMMARIZER(best_match_text, max_length=100, min_length=30, do_sample=False)
        response = summary[0]['summary_text']
    except Exception as e:
        print(f"⚠️ Summarization Error: {str(e)}")
        response = best_match_text
    
    print(f"✅ Response: {response}")
    return {"query": q, "response": response}

@app.get("/refresh/")
def refresh_docs():
    """Manually refresh the documentation index."""
    INDEX.reset()
    DOC_CHUNKS.clear()
    load_and_index_documents()
    return {"status": "Documentation refreshed successfully"}

# Load docs at startup
load_and_index_documents()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
