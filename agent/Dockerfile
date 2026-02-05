# Use an official Python image
#FROM python:3.9-slim
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

RUN chmod +x /app/agent.sh

# Expose the port on which Streamlit will run
EXPOSE 8502

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/agent.py", "--server.port=8502", "--server.address=0.0.0.0"]
