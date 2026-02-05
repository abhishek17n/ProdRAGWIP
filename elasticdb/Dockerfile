
# Use the official Elasticsearch image from DockerHub
FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.3

# Expose Elasticsearch ports
EXPOSE 9200 9300

# Set environment variables for single-node configuration
ENV discovery.type=single-node

# Set JVM options (adjust these based on your system)
ENV ES_JAVA_OPTS="-Xms512m -Xmx512m"

# Optional: Disable security for local development (not recommended for production)
ENV xpack.security.enabled=false

# Set property to allow the use of a dense vector field (for version 8.x and above)
ENV xpack.ml.enabled=false

# Specify the data directory for persistence
VOLUME /usr/share/elasticsearch/data


# No need to specify CMD, as the default entrypoint is already set correctly in the base image
