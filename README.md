# rag_mini: RAG from scratch in ~100 lines

    pip install -r requirements.txt
    python rag.py "Can I work from another country?"
    export ANTHROPIC_API_KEY=...   # optional, enables real answers

## Experiments (this is where the learning happens)
1. Ask something NOT in the docs. Does it say "I don't know"?
2. Change CHUNK_WORDS to 10, then 300. Watch retrieval quality change.
3. Set TOP_K=1, then 8. What breaks?
4. Ask "Can I work from abroad?" (no word overlap with "another country"). Compare the TF-IDF fallback vs sentence-transformers.
5. Add your own .txt files to data/.
6. Remove the "ONLY the context" instruction from the prompt and see hallucinations return.
