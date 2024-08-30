import streamlit as st
import os
from utils import (
    create_index,
    answer_query
)
import os
import shutil
import atexit

documents_folder_path = "sources"
vector_db_path = "vector_db"

def main():
    st.set_page_config(
        page_title="AI-Powered Document Intelligence",
        page_icon=":bulb:"  # Lightbulb icon
    )

    st.title("AI Assistant for State-of-the-Art Document Writing :books:")
    st.markdown("Unlock the knowledge within your documents. Ask questions, get insights, and craft documents with AI assistance.")

    user_question = st.text_input("Enter your query or request:")

    with st.sidebar:
        st.subheader("Your Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload  research papers or any relevant documents.", 
            accept_multiple_files=True, type=['pdf','doc','txt'] 
        )

        if st.button("Analyze"):
            if uploaded_files:
                save_uploaded_files(uploaded_files)
                with st.spinner("Documents uploaded and analysis in progress..."):
                    create_index(documents_folder_path, vector_db_path)
                st.success("Analysis done")


    chat_container = st.empty()  # Create a dynamic area for chat history
    if user_question:
        response = answer_query(user_question,vector_db_path)
        chat_container.markdown(f"**You:** {user_question}")  # Display user's query
        chat_container.markdown(f"**AI Assistant:** {response}")  # Display the answer from the AI

def cleanup_folders():
    folders = ['vector_db', 'sources']
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)  # Remove the entire folder and its contents
                print(f"Deleted folder: {folder}")
            except Exception as e:
                print(f"Failed to delete {folder}. Reason: {e}")
        
        try:
            os.makedirs(folder)  # Create a new empty folder
            print(f"Created new empty folder: {folder}")
        except Exception as e:
            print(f"Failed to create folder {folder}. Reason: {e}")


def save_uploaded_files(uploaded_files):
    # Define the path to the folder where files will be saved
    save_path = 'sources'

    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save each file to the specified folder
    for uploaded_file in uploaded_files:
        with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

if __name__ == '__main__':
    
    main()
