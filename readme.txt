docker build -t embedding-service .

docker run -p 8001:8001 embedding-service

http://0.0.0.0:8001

curl -X POST "http://localhost:8001/embed" \
-H "Content-Type: application/json" \
-d '{"query": "What is retrieval-augmented generation?"}'

python -c "import numpy; print(numpy.__version__)"

python3 -m venv venv

source venv/bin/activate
crewai==0.74.1
crewai[tools]