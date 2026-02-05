docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  elasticdb

docker build -t elasticdb .

python3 -m venv venv

source venv/bin/activate



curl http://localhost:9200

docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.0
