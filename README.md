# DocuMatic (Semantic Search Engine)

Semantic Search Engine - a document retrieval system to fetch relevant research papers from a Vector Database.

I decided to use 📚 hashtag#Haystack by deepset which is an amazing LLM framework for building (not limited to) semantic search systems. Haystack provides various easy & ready to use components to turn them into efficient pipelines in-order to quickly build LLM apps. My choice to use haystack is because it is optimized for search, document Q&A, and knowledge graphs.

Key aspects of the project: 
📄Scraped 300,000 papers from arXiv & Semantic Scholars on topics of Machine Learning, Deep Learning, Mathematics, Statistics & more.
🛢Decided to stick with haystack's native vector database for this simplicity purpose.
🤗Experimented with 3 different SOTA Sentence Transformer models 🚀 to generate word embeddings.
🔎Focusses on hybrid retrieval that makes use of BM25 for keyword or full-text search and Vector Embeddings for semantic search, & re-rank results with bge-reranker-base
⛵Used Streamlit to design the UI
