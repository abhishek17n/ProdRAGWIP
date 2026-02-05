docker build -t elastic_streamlit .

docker run -p 8501:8501 elastic_streamlit

To list all the indices in your Elasticsearch instance:
curl -X GET "http://localhost:9200/_cat/indices?v"

Check Data in a Specific Index:
To query documents from a specific index, 
you can use the _search endpoint. Here’s how you can retrieve data from an index 
(e.g., documents):
curl -X GET "http://localhost:9200/documents/_search?pretty"

Check a Specific Document by ID:

If you know the ID of a specific document and want to retrieve it, use this:
curl -X GET "http://localhost:9200/documents/_doc/{document_id}?pretty"


python3 -m venv venv

source venv/bin/activate