from elasticsearch import Elasticsearch
import logging
logging.basicConfig(level=logging.ERROR)

# Initialize Elasticsearch client
# Change to your Elasticsearch URL if needed
#es = Elasticsearch(hosts=["http://172.17.0.1:9200"])
es = Elasticsearch(hosts=["http://elasticdb:9200"])
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


# Function to index a document
def index_document(index_name, document_id, chunk_id, filename, chunk, embedding):
    doc = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "filename": filename,
        "chunk": chunk,
        "embedding": embedding
    }
    try:
        response = es.index(index=index_name, id=chunk_id, body=doc)
        print(f"Document with chunk ID '{chunk_id}' indexed: {response}")
    except Exception as e:
        print(f"Error indexing document: {str(e)}")

# Create the index
create_index(index_name, settings, mappings)

# Example: Index a document with embedding
example_document_id = "doc1"
example_chunk_id = "doc1_chunk_0"
example_filename = "example.txt"
example_chunk = "This is an example chunk of the document."
example_embedding = [0.01] * 768  # Placeholder embedding with 768 dimensions

# Index the document
index_document(index_name, example_document_id, example_chunk_id, example_filename, example_chunk, example_embedding)
