import streamlit as st
import logging
import torch
import os

from utils import haystack
from utils import ui

torch.classes.__path__ = []

def main():
    # Initialize the document store
    if 'document_store' not in st.session_state:
        with st.spinner("Loading Document Store..."):
            logging.debug("Document store not found in session state. Loading...")
            st.session_state.document_store = haystack.store_loader()
            print("DEBUG: Store Object: ", st.session_state.document_store)
            print("DEBUG: Document Count :", st.session_state.document_store.count_documents())
    else:
        logging.debug("Document store already found in session state. Using cached instance...")
    #with st.spinner("Loading Document Store..."):
        #document_store = haystack.store_loader()
        #print("DEBUG: Store Object: ", document_store)
        #print("DEBUG: Document Count :", document_store.count_documents())
    document_store = st.session_state.document_store

    # Initialize the pipeline (BAAI/bge-small-en-v1.5)
    pipeline = haystack.hybrid_retrieval_pipeline('sentence-transformers/msmarco-distilbert-base-v2', document_store, 5)

    # Run the UI
    #with st.spinner("Loading UI..."):
    ui.main(pipeline, document_store)

if __name__ == "__main__":
    main()