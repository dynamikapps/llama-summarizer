#!/bin/bash

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Warning: Ollama is not running. Starting Streamlit anyway, but you'll need to start Ollama separately."
fi

# Start the Streamlit app
echo "Starting Llama Summarizer Streamlit App..."
streamlit run streamlit_app.py 