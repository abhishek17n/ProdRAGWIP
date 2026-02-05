# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

RUN chmod +x /app/start_streamlit.sh

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app/streamlit.py", "--server.port=8501", "--server.enableCORS=false"]

# Command to run the Streamlit app
#CMD ["streamlit", "run", "app/elastic_search.py", "--server.port=8501", "--server.address=0.0.0.0"]