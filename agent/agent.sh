#!/bin/sh

# Introduce a delay to ensure all dependencies are up
sleep 40

# Command to run the Streamlit app
streamlit run app/agent.py --server.port=8502 --server.enableCORS=false