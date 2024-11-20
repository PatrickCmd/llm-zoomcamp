import os

import streamlit as st
# loading the OpenAI api key from .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from utils import load_document, chunk_data, create_embeddings, ask_and_get_answer, calculate_embedding_cost


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        st.write("History has been reset")


def main():
    st.image("img.png")
    st.subheader("LLM Question-Answering Application 🤖")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input("k", min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking, and embedding file"):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./files", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)
                
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: ${embedding_cost: .4f}")

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success("File uploaded, chunked and embedded successfully.")
    
    q = st.text_input("Ask a question about the content of your file:")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"K: {k}")
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer: ", value=answer)
    
            st.divider()
            if "history" not in st.session_state:
                st.session_state.history = ""
            value = f"Q: {q} \nA: {answer}"
            st.session_state.history = f"{value} \n {'-' * 100} \n {st.session_state.history}"
            h = st.session_state.history
            st.text_area(label="Chat History", value=h, key="history", height=400)


if __name__ == "__main__":
    main()
