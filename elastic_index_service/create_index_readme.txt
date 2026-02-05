docker build -t create_index_service .

docker run --network="host" create_index_service 

python3 -m venv venv

source venv/bin/activate

curl -X GET "http://localhost:9200/_cat/indices?v"