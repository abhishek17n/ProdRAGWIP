docker build -t embedding-service .

docker run -p 8001:8001 embedding-service

python3 -m venv venv

source venv/bin/activate