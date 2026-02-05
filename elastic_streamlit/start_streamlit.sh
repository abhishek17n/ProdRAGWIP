#!/bin/sh

# Introduce a delay to ensure all dependencies are up
sleep 30

# Command to run the Streamlit app
streamlit run app/streamlit.py --server.port=8501 --server.enableCORS=false
