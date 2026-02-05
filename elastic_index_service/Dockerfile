# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .

RUN chmod +x /app/start_index_creation.sh


# Set the default command to run the Python script
CMD ["python", "create_index.py"]