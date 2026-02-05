import os
import requests
import json
import streamlit as st
from elasticsearch import Elasticsearch
import openai
from utils import * 
from dotenv import load_dotenv
import asyncio
import fitz  # Import PyMuPDF
from datetime import datetime  # 👈 Added for timestamps

# Load environment variables from the .env file
load_dotenv()

# Initialize session state for Q&A history
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Custom CSS to style the sidebar and main content
st.markdown(
    """
    <style>
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7f7f7; /* Light grey for sidebar */
        padding: 10px;
    }

    /* Style the main content */
    .main-content {
        background-color: #ffffff; /* White for main content */
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Adjust the font and alignment */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #333333; /* Darker text in sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI Title
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.title("Document Chunk and Embedding Uploader/Searcher")

# Sidebar for File Upload
st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose files to upload", 
    accept_multiple_files=True
)

# Sidebar Process Button
if uploaded_files:
    st.sidebar.write("### Uploaded Files:")
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"- {uploaded_file.name}")

    if st.sidebar.button("Process and Upload All Files"):
        processed_files = []
        failed_files = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read file content
                document_content = ""
                st.write(f"- {uploaded_file.type }")
                # Check if the file is a PDF
                if uploaded_file.type == "application/pdf":
                    # Use PyMuPDF to extract text from PDF
                    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document[page_num]
                        document_content += page.get_text()
                    st.write(f"- {document_content}")
                    pdf_document.close()
                else:
                    try:
                        document_content = uploaded_file.read().decode("utf-8").strip()
                        st.write(f"- {document_content}")
                    except UnicodeDecodeError:
                        document_content = uploaded_file.read().decode("latin-1").strip()

                document_id = uploaded_file.name  # Using filename as document ID

                # Generate embeddings and index the document
                st.write(f"Processing '{uploaded_file.name}'...")
                index_document_with_embedding(document_id, uploaded_file.name, document_content)
                st.success(f"File '{uploaded_file.name}' successfully processed and indexed.")
                processed_files.append(uploaded_file.name)

            except Exception as e:
                st.error(f"Failed to process file '{uploaded_file.name}': {str(e)}")
                failed_files.append(uploaded_file.name)

        # Show a summary
        st.write("### Processing Summary:")
        if processed_files:
            st.write("#### Successfully Processed Files:")
            for file in processed_files:
                st.write(f"- {file}")
        if failed_files:
            st.write("#### Failed Files:")
            for file in failed_files:
                st.write(f"- {file}")

else:
    st.sidebar.info("No files uploaded yet.")

# Sidebar: Display distinct filenames as checkboxes
with st.sidebar:
    try:
        filenames = get_distinct_filenames(
            index_name="documents",
            scroll_time="2m",
            batch_size=1000
        )
        st.info(f"Distinct files: {filenames}")
    except Exception as e:
        st.warning(f"Could not fetch filenames: {str(e)}")
        filenames = []

    selected_filenames = []
    if filenames:
        for filename in filenames:
            if st.checkbox(filename, key=f"file_{filename}"):
                selected_filenames.append(filename)
    else:
        st.sidebar.warning("No filenames found in the index.")

    st.write("### Selected Filenames")
    if selected_filenames:
        for filename in selected_filenames:
            st.write(f"- {filename}")
    else:
        st.write("No filenames selected.")

# Main Page for Search
st.header("Search Documents")
query = st.text_input("Enter your search query")
genaioption = st.selectbox(
    'Gen AI Option?',
    ('Gemini', 'OpenAI', 'Ollama')
)
st.write('You selected:', genaioption)

# Async helper function
async def my_async_function(query):
    answers = response(query)
    await asyncio.sleep(0.1)  # Just to simulate async
    return answers

# Search button
if st.button("Search"):
    if query:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answers = loop.run_until_complete(my_async_function(query))

        # Filter results by selected filenames (if any)
        if selected_filenames:
            results = filter_by_filenames(selected_filenames, "documents")
        else:
            results = search_documents(query)  # or however you get all results

        # Prepare Q&A entry for history
        qa_entry = {
            "question": query,
            "answer": answers,
            "genaioption": genaioption,
            "selected_files": selected_filenames.copy(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Elasticsearch" if results else genaioption,
            "results": []
        }

        if results:
            st.write("### Answer:")
            st.write(answers)
            st.write("### Search Results:")
            for result in results:
                st.write(f"**Document ID:** {result['_source']['document_id']}")
                st.write(f"**Filename:** {result['_source']['filename']}")
                st.write(f"**Chunk:** {result['_source']['chunk']}")
                st.write("---")
                # Store in history
                qa_entry["results"].append({
                    "document_id": result['_source']['document_id'],
                    "filename": result['_source']['filename'],
                    "chunk": result['_source']['chunk'][:300] + "..." if len(result['_source']['chunk']) > 300 else result['_source']['chunk']
                })
        else:
            st.warning(f"No results found in Elasticsearch. Generating response from {genaioption}...")
            if genaioption == 'OpenAI':
                openai_response = get_response_from_openai(query)
                st.write(f"### OpenAI Response: {openai_response}")
                qa_entry["answer"] = openai_response
            else:
                gemini_response = get_response_from_gemini(query)
                st.write(f"### Gemini Response: {gemini_response}")
                qa_entry["answer"] = gemini_response

        # 👇 SAVE TO HISTORY
        st.session_state.qa_history.append(qa_entry)
        st.success("✅ Q&A saved to history!")

    else:
        st.error("Please enter a query to search.")

# Create the Elasticsearch index (if not exists)
try:
    create_index(index_name, settings, mappings)
except Exception as e:
    st.error(f"Failed to create Elasticsearch index: {str(e)}")

# 👇 DISPLAY Q&A HISTORY
if st.session_state.qa_history:
    st.markdown("---")
    st.title("📚 Q&A History")
    for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(f"Q{i}: {qa['question'][:60]}{'...' if len(qa['question']) > 60 else ''} ({qa['timestamp']})"):
            st.markdown(f"**Asked at**: {qa['timestamp']}")
            st.markdown(f"**Model/Source**: `{qa['source']}`")
            if qa['selected_files']:
                st.markdown(f"**Filtered Files**: `{', '.join(qa['selected_files'])}`")
            st.markdown(f"**Answer**: {qa['answer']}")

            if qa.get("results"):
                st.markdown("**Retrieved Chunks:**")
                for idx, r in enumerate(qa["results"]):
                    st.markdown(f"- **File**: `{r['filename']}` (ID: {r['document_id']})")
                    # 👇 UNIQUE KEY using question index + chunk index
                    st.text_area(
                        label="Chunk Preview",
                        value=r['chunk'],
                        height=100,
                        disabled=True,
                        key=f"chunk_{i}_{idx}"  # Unique per Q&A and per chunk
                    )
                    st.markdown("---")
            # if qa.get("results"):
            #     st.markdown("**Retrieved Chunks:**")
            #     for r in qa["results"]:
            #         st.markdown(f"- **File**: `{r['filename']}` (ID: {r['document_id']})")
            #         st.text_area("Chunk Preview", r['chunk'], height=100, disabled=True)
            #         st.markdown("---")

# Close div
st.markdown('</div>', unsafe_allow_html=True)