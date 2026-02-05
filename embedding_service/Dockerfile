# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that the embedding service will run on
EXPOSE 8001

# Run the service
CMD ["uvicorn", "embedding_service:app", "--host", "0.0.0.0", "--port", "8001"]
