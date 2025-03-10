#!/bin/bash

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Check if the Llama 3 model is available
if ! curl -s http://localhost:11434/api/tags | grep -q "llama3:8b"; then
    echo "Warning: llama3:8b model not found. Attempting to pull it now..."
    ollama pull llama3:8b
fi

# Check for command line argument for input file
if [ "$1" != "" ]; then
    INPUT_FILE="$1"
else
    # Default to sample markdown document
    INPUT_FILE="sample_document.md"
fi

# Generate output filename based on input
FILENAME=$(basename -- "$INPUT_FILE")
EXTENSION="${FILENAME##*.}"
FILENAME="${FILENAME%.*}"
OUTPUT_FILE="${FILENAME}_summary.md"

# Run the summarizer on the specified document
echo "Running summarizer on $INPUT_FILE..."
python summarize.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --max_tokens 1000 --chunk_size 4000 --overlap 500

echo "Done! Summary saved to $OUTPUT_FILE"
echo ""
echo "Usage examples:"
echo "  ./run_summarizer.sh document.md    # Summarize a markdown file"
echo "  ./run_summarizer.sh document.pdf   # Summarize a PDF file"
echo "  ./run_summarizer.sh document.docx  # Summarize a Word document" 