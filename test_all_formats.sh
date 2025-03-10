#!/bin/bash

# This script demonstrates summarization of different file formats
# It converts the sample markdown file to different formats for testing

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

echo "=== Llama Summarizer Format Test ==="
echo "This script tests summarization of different file formats"
echo ""

# Check for required tools
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed. It's needed to convert between formats."
    echo "Install it with: brew install pandoc"
    exit 1
fi

# If we're testing PDF, check for wkhtmltopdf
if ! command -v wkhtmltopdf &> /dev/null; then
    echo "Note: wkhtmltopdf is not installed. PDF test will be skipped."
    echo "Install it with: brew install wkhtmltopdf"
    TEST_PDF=false
else
    TEST_PDF=true
fi

echo "1. Testing markdown summarization..."
./run_summarizer.sh sample_document.md
echo ""

echo "2. Converting to HTML and testing HTML summarization..."
pandoc sample_document.md -o sample_document.html
./run_summarizer.sh sample_document.html
echo ""

echo "3. Converting to DOCX and testing Word document summarization..."
pandoc sample_document.md -o sample_document.docx
./run_summarizer.sh sample_document.docx
echo ""

if [ "$TEST_PDF" = true ]; then
    echo "4. Converting to PDF and testing PDF summarization..."
    wkhtmltopdf sample_document.html sample_document.pdf
    ./run_summarizer.sh sample_document.pdf
    echo ""
fi

echo "=== Test Complete ==="
echo "Check the generated summary files:"
echo "- sample_document_summary.md (from Markdown)"
echo "- sample_document_summary.md (from HTML)"
echo "- sample_document_summary.md (from Word)"
if [ "$TEST_PDF" = true ]; then
    echo "- sample_document_summary.md (from PDF)"
fi 