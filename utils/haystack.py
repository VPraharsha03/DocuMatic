import streamlit as st

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker

import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
store_path = os.path.join(current_dir, 'store', 'document-store-msmarco-distilroberta-base-v2.json')
print("DEBUG Store PATH: ", store_path)

meta = ["title", "authors", "pub_date", "category", "source"]

@st.cache_resource(show_spinner=False)
def indexing(embedding_model: str, chunk_size: int):
    document_cleaner = DocumentCleaner()
    
    document_splitter = DocumentSplitter(split_by="word", split_length=chunk_size)
    
    document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model, meta_fields_to_embed=meta, device=ComponentDevice.from_str("cuda:0")
        )
    
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)
    #pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
    #indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)
    #pipeline.connect("converter", "cleaner")
    indexing_pipeline.connect("cleaner", "embedder")
    #indexing_pipeline.connect("converter", "writer")
    #indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")
    #pipeline.connect("cleaner", "splitter")
    #pipeline.connect("splitter", "writer")
    
    #indexing_pipeline.run({"converter": {"sources": ["/kaggle/input/s2orc-arxiv-scraped/combined_cleaned.csv"]}})
    indexing_pipeline.run(#{"splitter": {"documents": docs}}
                          {"cleaner": {"documents": docs_valid}}
                         )

    return document_store

@st.cache_data()
def store_loader():
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    doc_store = doc_store.load_from_disk(store_path)
    return doc_store

@st.cache_resource(show_spinner=False)
def hybrid_retrieval_pipeline(embedding_model: str, _document_store, top_k=5):
    text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model, device=ComponentDevice.from_str("cuda:0")
    )
    embedding_retriever = InMemoryEmbeddingRetriever(_document_store, top_k=top_k)
    bm25_retriever = InMemoryBM25Retriever(_document_store, top_k=top_k)
    
    document_joiner = DocumentJoiner()
    
    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")
    
    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("document_joiner", document_joiner)
    hybrid_retrieval.add_component("ranker", ranker)
    
    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "document_joiner")
    hybrid_retrieval.connect("embedding_retriever", "document_joiner")
    hybrid_retrieval.connect("document_joiner", "ranker")

    return hybrid_retrieval