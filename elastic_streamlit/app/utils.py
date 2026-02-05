import os
import requests
import json
import streamlit as st
from elasticsearch import Elasticsearch
import openai
import asyncio
from dotenv import load_dotenv
import logging
os.environ["LANGCHAIN_VERBOSE"] = "1"


logging.basicConfig(level=logging.DEBUG)
# Load environment variables from the .env file
load_dotenv()

# Embedding Service, Elasticsearch URLs, and OpenAI API Key
#EMBEDDING_SERVICE_URL = "http://172.17.0.4:8001/embed"
#ELASTICSEARCH_URL = "http://172.17.0.1:9200/documents/_doc" 
EMBEDDING_SERVICE_URL = "http://embedding_service:8001/embed"
ELASTICSEARCH_URL = "http://elasticdb:9200/documents/_doc" 
 # Replace 'documents' with your index name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")

# Initialize Elasticsearch client
es = Elasticsearch(hosts=["http://elasticdb:9200"])  # Change to your Elasticsearch URL if needed

# Define the index settings and mappings
index_name = 'documents'
settings = {
    "number_of_shards": 1,
    "number_of_replicas": 1
}

# Adding new fields: document_id, chunk_id, filename, chunk, and embedding
mappings = {
    "properties": {
        "document_id": {
            "type": "keyword"  # keyword for exact match
        },
        "chunk_id": {
            "type": "keyword"  # keyword for exact match
        },
        "filename": {
            "type": "text"  # text for full-text search capability
        },
        "chunk": {
            "type": "text"  # text for the chunk of the document
        },
        "embedding": {
            "type": "dense_vector",  # use dense vector for storing embeddings
            "dims": 768              # assuming embeddings have 768 dimensions
        }
    }
}

# Function to create an index
def create_index(index_name, settings, mappings):
    if not es.indices.exists(index=index_name):
        response = es.indices.create(
            index=index_name,
            body={
                "settings": settings,
                "mappings": mappings
            }
        )
        print(f"Index '{index_name}' created: {response}")
    else:
        print(f"Index '{index_name}' already exists")

# Function to generate embedding for a document using embedding service
def get_embedding_from_service(document):
    """Send a document to the embedding service to generate an embedding."""
    try:
        # Truncate the document if it exceeds the model's maximum sequence length
        max_length = 512
        truncated_document = document[:max_length]
        
        response = requests.post(EMBEDDING_SERVICE_URL, json={"query": truncated_document})
        
        if response.status_code == 200:
            # Return the embedding from the service response
            return response.json()["embedding"]
        else:
            raise Exception(f"Error in embedding service: {response.status_code}, {response.text}")
    
    except Exception as e:
        raise Exception(f"Failed to get embedding for document: {document}. Error: {str(e)}")

# Function to chunk text
import re

def chunk_text(text, min_chunk_size=400, max_chunk_size=500):
    """Split the document into chunks of a given size, ensuring words and sentences are not split."""
    # Split the text into sentences using a regular expression
    sentences = re.split(r'(?<=[.!?]) +', text)
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Ensure the sentence itself isn't longer than the max chunk size
        words = sentence.split()
        for word in words:
            # Check if adding the current word would exceed the max chunk size
            if current_length + len(word) + 1 > max_chunk_size:
                # Only yield the chunk if it meets the minimum size
                if current_length >= min_chunk_size:
                    yield ' '.join(current_chunk)
                    current_chunk = []
                    current_length = 0
                else:
                    break  # Skip splitting the sentence if it breaks the minimum size condition

            # Add the word to the current chunk
            current_chunk.append(word)
            current_length += len(word) + 1  # Account for the space between words

        # Check if we need to start a new chunk after adding the sentence
        if current_length >= min_chunk_size and current_chunk:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0

    # Yield any remaining words in the final chunk if it meets the minimum size
    if current_chunk and current_length >= min_chunk_size:
        yield ' '.join(current_chunk)

# Function to index a document along with its embedding and chunks
def index_document_with_embedding(document_id, filename, document_content):
    # Step 2: Chunk the document and index each chunk with its own embedding
    chunks = list(chunk_text(document_content))
    
    for chunk_num, chunk in enumerate(chunks):
        # Get embedding for each chunk
        embedding = get_embedding_from_service(chunk)
        
        # Ensure embedding is formatted correctly (list of floats)
        if not isinstance(embedding, list):
            raise ValueError(f"Invalid embedding format for chunk {chunk_num}")

        doc = {
            "document_id": document_id,
            "chunk_id": f"{document_id}_chunk_{chunk_num}",
            "filename": filename,
            "chunk": chunk,
            "embedding": embedding
        }

        # Use the appropriate argument based on your Elasticsearch Python client version
        response = es.index(index=index_name, id=f"{document_id}_chunk_{chunk_num}", document=doc)

        # Print a more concise response message
        print(f"Document with chunk ID '{document_id}_chunk_{chunk_num}' indexed: {response['result']}")

# Function to search for documents in Elasticsearch

# Step 3: Vector Index setup with Elasticsearch 

def search_documents(query):
    """Search for documents in Elasticsearch based on the query."""
    
    try:
        # Generate embedding for the query using embedding service
        embedding = get_embedding_from_service(query)
        
        # Define the search query with a combination of cosine similarity and keyword matching
        # search_body = {
        #     "query": {
        #         "bool": {
        #             "must": [
        #                 {
        #                     "script_score": {
        #                         "query": {"match_all": {}},
        #                         "script": {
        #                             "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        #                             "params": {"query_vector": embedding}
        #                         }
        #                     }
        #                 },
        #                 {
        #                     "match": {"chunk": query}
        #                 }
        #             ]
        #         }
        #     }
        # }
        
        search_body = {
  "query": {
    "bool": {
      "must": [
        {
          "script_score": {
            "query": {"match_all": {}},
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
              "params": {"query_vector": embedding}
            }
          }
        },
        {
          "match": {
            "chunk": {
              "query": query,
              "boost": 2.0
            }
          }
        }
      ],
      "should": [
        {
          "match": {
            "title": {
              "query": query,
              "boost": 3.0
            }
          }
        }
      ]
    }
  }
}

        # Perform the search
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        if hits:
            return hits
        else:
            raise Exception("No results found in Elasticsearch")
    except Exception as e:
        st.error(f"An error occurred while searching: {str(e)}")
        return []

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from openai import OpenAI
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def response1(query):
    try:
        # Search for documents related to the query
        results = search_documents(query)
        # Prepare the context from the search results
        context = "\n".join([result['_source']['chunk'] for result in results])

        # Prepare the prompt template
        review_template = """
        Your job is to use documents to answer questions.
        Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context.
        If you don't know an answer, say you don't know. 
        Context: {context}
        """
        
        # Set up the prompt templates for system and user input
        review_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["context"], template=review_template)
        )
        review_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["query"], template="{query}")
        )
        
        # Combine the messages
        messages = [review_system_prompt, review_human_prompt]
        review_prompt = ChatPromptTemplate(input_variables=["context", "query"], messages=messages)

        

        # Use OpenAI directly for response generation
        prompt = review_prompt.format(context=context, query=query)
        
        client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
           api_key=OPENAI_API_KEY
        )   
        # Using ChatCompletion API with the newer model
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": review_template.format(context=context)},
            {"role": "user", "content": prompt}],
            max_tokens=150
        )
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    return response.choices[0].message.content.strip()


def response(query):
    try:
        # Search for documents related to the query
        results = search_documents(query)
        # Prepare the context from the search results
        context = "\n".join([result['_source']['chunk'] for result in results])

        # Initialize LangChain Agent with Gemini
          # The client automatically picks up the API key from the environment variable
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash-lite')

        # llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash",
        #     temperature=0.5,
        #     max_tokens=None,
        #     timeout=None,
        #     google_api_key=google_api_key,
        #     verbose=True  # Enable verbose logging
        # )
        prompt = f"Question: {query}\nContext: {context}\nProvide a helpful answer based on the given context."
        response = model.generate_content([prompt])

        return response.text

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# Function to get a response from OpenAI if no results found in Elasticsearch
def get_response_from_gemini(query, context=""):
    """Generate a response from Gemini if no results are found in Elasticsearch."""
    try:
        prompt = f"Question: {query}\nContext: {context}\nProvide a helpful answer based on the given context."

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        st.error(f"An error occurred while getting a response from Gemini: {str(e)}")
        return ""
    return response.content

# Function to get a response from OpenAI if no results found in Elasticsearch
def get_response_from_openai(query, context=""):
    """Generate a response from OpenAI if no results are found in Elasticsearch."""
    try:
        prompt = f"Question: {query}\nContext: {context}\nProvide a helpful answer based on the given context."
        client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
           api_key=OPENAI_API_KEY
        )   
        # Using ChatCompletion API with the newer model
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
            max_tokens=150
    )
    except Exception as e:
        st.error(f"An error occurred while getting a response from OpenAI: {str(e)}")
        return ""
    return response.choices[0].message.content.strip()

# Streamlit UI to upload file and process chunks
st.title("Document Chunk and Embedding Uploader/Searcher")

uploaded_file = st.file_uploader("Choose a text file")

if uploaded_file is not None:
    try:
        document_content = uploaded_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        document_content = uploaded_file.read().decode("latin-1").strip()

    document_id = uploaded_file.name  # Using filename as document ID

    # Step 1: Generate Embeddings and Index the Document
    if st.button("Process and Upload"):
        try:
            st.write("Generating embeddings and indexing the document...")
            index_document_with_embedding(document_id, uploaded_file.name, document_content)
            st.success(f"File '{uploaded_file.name}' successfully processed and indexed.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
from typing import List, Set

def get_distinct_filenames(
                          index_name: str = "documents", 
                          scroll_time: str = "2m", 
                          batch_size: int = 1000) -> List[str]:
    """
    Retrieve a list of distinct filenames from an Elasticsearch index using the Scroll API.
    
    Args:
        es_host (str): Elasticsearch host URL (default: "http://localhost:9200").
        index_name (str): Name of the Elasticsearch index.
        scroll_time (str): Scroll context duration (e.g., "2m" for 2 minutes).
        batch_size (int): Number of documents to fetch per scroll request.
    
    Returns:
        List[str]: A list of unique filename values.
    
    Raises:
        Exception: If there's an error connecting to Elasticsearch or executing the query.
    """
    try:
        # Initialize Elasticsearch client
        
        # Check if the client is connected
        if not es.ping():
            raise Exception("Failed to connect to Elasticsearch at " + es_host)
        
        # Initialize set to store unique filenames
        unique_filenames: Set[str] = set()
        
        # Initial scroll query
        query = {
            "query": {"match_all": {}},
            "size": batch_size
        }
        response = es.search(index=index_name, body=query, scroll=scroll_time)
        scroll_id = response["_scroll_id"]
        
        # Process first batch
        for hit in response["hits"]["hits"]:
            if "filename" in hit["_source"]:
                unique_filenames.add(hit["_source"]["filename"])
        
        # Scroll through remaining batches
        while len(response["hits"]["hits"]):
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            for hit in response["hits"]["hits"]:
                if "filename" in hit["_source"]:
                    unique_filenames.add(hit["_source"]["filename"])
        
        # Clear scroll context to free resources
        es.clear_scroll(scroll_id=scroll_id)
        
        # Convert set to sorted list for consistent output
        return sorted(list(unique_filenames))
    
    except Exception as e:
        print(f"Error retrieving distinct filenames: {str(e)}")
        return []
        
def filter_by_filenames(selected_filenames: List[str],  
                       index_name: str = "documents") -> List[dict]:
    """
    Filter documents from an Elasticsearch index by a list of selected filenames.
    
    Args:
        selected_filenames (List[str]): List of filenames to filter by.
        es_host (str): Elasticsearch host URL (default: "http://localhost:9200").
        index_name (str): Name of the Elasticsearch index.
    
    Returns:
        List[dict]: List of Elasticsearch hits (documents) matching the selected filenames.
    
    Raises:
        Exception: If there's an error connecting to Elasticsearch or executing the query.
    """
    try:
        # Initialize Elasticsearch client

        if not es.ping():
            raise Exception("Failed to connect to Elasticsearch at " + es_host)
        
        # If no filenames are selected, return an empty list
        if not selected_filenames:
            return []
        
        # Build the query with a bool should clause for multiple filenames
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"filename": filename}} for filename in selected_filenames
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 1000  # Adjust based on expected number of results
        }
        
        # Perform the search
        response = es.search(index=index_name, body=query)
        hits = response["hits"]["hits"]
        
        if hits:
            return hits
        else:
            st.warning("No documents found for the selected filenames.")
            return []
    
    except Exception as e:
        st.error(f"Error filtering documents by filenames: {str(e)}")
        return []