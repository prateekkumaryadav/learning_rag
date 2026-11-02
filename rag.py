"""
Minimal RAG from scratch. No frameworks, so every step is visible.

Pipeline:  
load -> chunk -> embed -> store   (indexing, done once)
question -> embed -> top-k search -> build prompt -> LLM   (querying)

Run:  python rag.py "How many days of annual leave do I get?"
"""
import glob, os, sys
import numpy as np

# size of each chunk
CHUNK_WORDS = 60   

# words shared between neighbouring chunks (keeps context)
OVERLAP     = 15

# how many chunks to retrieve
TOP_K       = 3    


# ---------- 1. LOAD ----------
def load_docs(folder=None):
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    paths = glob.glob(os.path.join(folder, "*.txt"))
    if not paths and folder.endswith(os.path.join("", "data")):
        paths = glob.glob(os.path.join(os.path.dirname(folder), "*.txt"))
    docs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in {folder} or the project root."
        )
    return docs


# ---------- 2. CHUNK ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


# ---------- 3. EMBED ----------
class Embedder:
    """Turns text into vectors. Uses a real embedding model if installed, otherwise falls back to TF-IDF (keyword-based, weaker, but offline)."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.name = "sentence-transformers (semantic)"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(stop_words="english")
            self.name = "TF-IDF fallback (keyword only)"
        self.is_tfidf = self.name.startswith("TF-IDF")

    def fit(self, chunks):
        if self.is_tfidf:
            self.model.fit(chunks)

    def encode(self, texts):
        if self.is_tfidf:
            vecs = self.model.transform(texts).toarray()
        else:
            vecs = self.model.encode(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)   # unit length => dot product = cosine similarity


# ---------- 4. STORE + 5. RETRIEVE ----------
class VectorStore:
    """The simplest possible vector DB: a numpy matrix."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks, self.sources = [], []

    def add(self, docs):
        for source, text in docs:
            for c in chunk_text(text):
                self.chunks.append(c)
                self.sources.append(source)
        self.embedder.fit(self.chunks)
        self.vectors = self.embedder.encode(self.chunks)

    def search(self, query, k=TOP_K):
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q                      # cosine similarity with every chunk
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], self.sources[i], float(scores[i])) for i in top]


# ---------- 6. AUGMENT (build the prompt) ----------
def build_prompt(question, hits):
    context = "\n\n".join(f"[{src}] {chunk}" for chunk, src, _ in hits)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I don't know\". "
        "Mention which source file you used.\n\n"
        f"### Context\n{context}\n\n### Question\n{question}\n\n### Answer"
    )


# ---------- 7. GENERATE ----------
# def generate(prompt):
#     if not os.getenv("ANTHROPIC_API_KEY"):
#         return None
#     import anthropic
#     client = anthropic.Anthropic()
#     resp = client.messages.create(
#         model="claude-sonnet-5-5",
#         max_tokens=500,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return resp.content[0].text


# def generate(prompt):
#     if not os.getenv("GROQ_API_KEY"):
#         return None
#     from groq import Groq

#     client = Groq(api_key=os.environ["GROQ_API_KEY"])
#     resp = client.chat.completions.create(
#         # model="llama-3.3-70b-versatile",
#         model="whisper-large-v3-turbo",
#         max_tokens=500,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return resp.choices[0].message.content

def generate(prompt):
    if not os.getenv("GROQ_API_KEY"):
        return None
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"

    embedder = Embedder()
    store = VectorStore(embedder)
    store.add(load_docs())
    print(f"Embedder: {embedder.name} | chunks indexed: {len(store.chunks)}\n")

    hits = store.search(question)
    print("RETRIEVED CHUNKS")
    for chunk, src, score in hits:
        print(f"  score={score:.3f}  [{src}]  {chunk[:90]}...")

    prompt = build_prompt(question, hits)
    answer = generate(prompt)
    if answer:
        print(f"\nANSWER\n{answer}")
    else:
        print("\nNo GROQ_API_KEY set, so here is the prompt an LLM would receive:\n")
        print(prompt)