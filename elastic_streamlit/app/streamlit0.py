# import os
# import requests
# import json
# import streamlit as st
# from elasticsearch import Elasticsearch
# import openai
# from utils import * 
# from dotenv import load_dotenv

# # Load environment variables from the .env file
# load_dotenv()

# # Streamlit UI Title
# st.title("Document Chunk and Embedding Uploader/Searcher")

# # Sidebar for File Upload
# st.sidebar.header("Upload Files")
# uploaded_files = st.sidebar.file_uploader(
#     "Choose files to upload", 
#     accept_multiple_files=True
# )

# # Sidebar Process Button
# if uploaded_files:
#     st.sidebar.write("### Uploaded Files:")
#     for uploaded_file in uploaded_files:
#         st.sidebar.write(f"- {uploaded_file.name}")

#     if st.sidebar.button("Process and Upload All Files"):
#         processed_files = []
#         failed_files = []
        
#         for uploaded_file in uploaded_files:
#             try:
#                 # Read file content
#                 try:
#                     document_content = uploaded_file.read().decode("utf-8").strip()
#                 except UnicodeDecodeError:
#                     document_content = uploaded_file.read().decode("latin-1").strip()

#                 document_id = uploaded_file.name  # Using filename as document ID

#                 # Generate embeddings and index the document
#                 st.write(f"Processing '{uploaded_file.name}'...")
#                 index_document_with_embedding(document_id, uploaded_file.name, document_content)
#                 st.success(f"File '{uploaded_file.name}' successfully processed and indexed.")
#                 processed_files.append(uploaded_file.name)

#             except Exception as e:
#                 st.error(f"Failed to process file '{uploaded_file.name}': {str(e)}")
#                 failed_files.append(uploaded_file.name)

#         # Show a summary
#         st.write("### Processing Summary:")
#         if processed_files:
#             st.write("#### Successfully Processed Files:")
#             for file in processed_files:
#                 st.write(f"- {file}")
#         if failed_files:
#             st.write("#### Failed Files:")
#             for file in failed_files:
#                 st.write(f"- {file}")

# else:
#     st.sidebar.info("No files uploaded yet.")

# # Main Page for Search
# st.header("Search Documents")
# query = st.text_input("Enter your search query")

# if st.button("Search"):
#     if query:
#         results = search_documents(query)
#         answers = response(query)
#         if results:
#             st.write("### Search Results:")
#             for result in results:
#                 st.write(f"**Document ID:** {result['_source']['document_id']}")
#                 st.write(f"**Filename:** {result['_source']['filename']}")
#                 st.write(f"**Chunk:** {result['_source']['chunk']}")
#                 st.write("---")
#         else:
#             st.warning("No results found in Elasticsearch. Generating response from OpenAI...")
#             openai_response = get_response_from_openai(query)
#             st.write(f"### OpenAI Response: {openai_response}")

#         # Display the response
#         st.write("### Answer:")
#         st.write(answers)
#     else:
#         st.error("Please enter a query to search.")

# # Create the Elasticsearch index
# try:
#     create_index(index_name, settings, mappings)
# except Exception as e:
#     st.error(f"Failed to create Elasticsearch index: {str(e)}")





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
# Load environment variables from the .env file
load_dotenv()

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
with st.sidebar:
    # Usage
    # index_name = 'documents'
    #Simple usage
    filenames = get_distinct_filenames(
        index_name="documents",
        scroll_time="2m",
        batch_size=1000
    )
    st.info(f"Distinct files: {filenames}")
# Main Page for Search
st.header("Search Documents")
query = st.text_input("Enter your search query")
genaioption = st.selectbox(
 'Gen AI Option?',
('Gemini', 'OpenAI', 'Ollama'))
st.write('You selected:', genaioption)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
async def my_async_function(query):
    # Your asynchronous code here
    answers = response(query)
    await asyncio.sleep(1)
    st.write("Async function completed!")
    return answers
if st.button("Search"):
    if query:
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = search_documents(query)
        # answers = response(query)
        answers = loop.run_until_complete(my_async_function(query))
        if results:
            # Display the response
            st.write("### Answer:")
            st.write(answers)
            st.write("### Search Results:")
            for result in results:
                st.write(f"**Document ID:** {result['_source']['document_id']}")
                st.write(f"**Filename:** {result['_source']['filename']}")
                st.write(f"**Chunk:** {result['_source']['chunk']}")
                st.write("---")
        else:
            st.warning(f"No results found in Elasticsearch. Generating response from OpenAI/Gemini...{genaioption}")
            if genaioption=='OpenAI':
                openai_response = get_response_from_openai(query)
                st.write(f"### OpenAI Response: {openai_response}")
            else:
                gemini_response = get_response_from_gemini(query)
                st.write(f"### Gemini Response: {gemini_response}")
        
    else:
        st.error("Please enter a query to search.")

# Create the Elasticsearch index
try:
    create_index(index_name, settings, mappings)
except Exception as e:
    st.error(f"Failed to create Elasticsearch index: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
