import streamlit as st
from PIL import Image

from dotenv import load_dotenv

# Set page title and layout
st.set_page_config(page_title="Semantic Search Engine", layout="wide")

def reset_results(*args):
    st.session_state.results = None

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def set_initial_state():
    load_dotenv()
    set_state_if_absent("query", "")
    set_state_if_absent("results", None)

def display_logo():
    # Load Haystack logo
    haystack_logo = Image.open("logo/haystack-logo.png")

    # Create header section
    st.header("")
    col1, col2, col3 = st.columns([1, 140, 1])
    with col1:
        st.write("")
    with col2:
        st.image(haystack_logo, width=200)
    with col3:
        st.write("")

def display_header():
    st.header("Semantic Search Engine")

def display_search_bar():
    search_query = st.text_input("Search for research papers...", placeholder="Type your query here", value=st.session_state.query, max_chars=150, on_change=reset_results)
    return search_query

def display_filters():
    col1, col2, col3 = st.columns([4, 4, 4])
    with col1:
        filter_by_date = st.checkbox("Filter by date")
    with col2:
        filter_by_source = st.checkbox("Filter by source")
    with col3:
        filter_by_category = st.checkbox("Filter by category")
    return filter_by_date, filter_by_source, filter_by_category

def display_search_button():
    if st.button("Search"):
        return True
    else:
        return False

def display_footer():
    st.write("Powered by Haystack")
    st.write(":heart: Made by Vivek Praharsha")

@st.cache_data(show_spinner=True)
def query(_pipeline, search_query):
    print("DEBUG: Pipeline will run with query:", search_query)
    with st.spinner("🔎 Searching for papers..."):
        results = _pipeline.run({"text_embedder": {"text": search_query}, "bm25_retriever": {"query": search_query}, "ranker": {"query": search_query}})
    return results

def fetch_results(pipeline, search_query):
    # Add your search logic here
    st.session_state.results = query(pipeline, search_query)
    results = st.session_state.results
    #results = pipeline.run({"text_embedder": {"text": search_query}, "bm25_retriever": {"query": search_query}, "ranker": {"query": search_query}})
    st.write("Search results:")
    for result in results["ranker"]["documents"]:
        st.write("------------------------------------------------------------------------------------------------")
        st.write("Title : ", result.meta["title"])
        st.write("Abstract : ", result.content)
        st.write("Source : ", result.meta["source"])
        st.write("------------------------------------------------------------------------------------------------")

def main(pipeline, document_store):
    set_initial_state()
    display_logo()
    display_header()
    search_query = display_search_bar()
    #print("DEBUG:", search_query)
    filter_by_date, filter_by_source, filter_by_category = display_filters()
    if display_search_button():
        # Add your search logic here
        fetch_results(pipeline, search_query)
        '''
        st.session_state.results = query(pipeline, search_query)
        results = st.session_state.results
        #results = pipeline.run({"text_embedder": {"text": search_query}, "bm25_retriever": {"query": search_query}, "ranker": {"query": search_query}})
        st.write("Search results:")
        for result in results["ranker"]["documents"]:
            st.write("------------------------------------------------------------------------------------------------")
            st.write("Title : ", result.meta["title"])
            st.write("Abstract : ", result.content)
            st.write("Source : ", result.meta["source"])
            st.write("------------------------------------------------------------------------------------------------")
        '''
    display_footer()

if __name__ == "__main__":
    main()