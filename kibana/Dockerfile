# Use the official Kibana image from DockerHub
FROM docker.elastic.co/kibana/kibana:8.15.3

# Expose Kibana's default port
EXPOSE 5601

# Set environment variables for Elasticsearch connection
ENV ELASTICSEARCH_HOSTS="http://elasticdb:9200"

# Optional: Set Kibana server host to allow external access
ENV SERVER_HOST=0.0.0.0

# Optional: Set the default space for Kibana (modify if needed)
ENV SERVER_BASEPATH=""

# Optional: Set xpack security for development (disable if required)
#ENV ELASTICSEARCH_USERNAME=elastic
#ENV ELASTICSEARCH_PASSWORD=changeme

# Run Kibana
CMD ["bin/kibana"]
