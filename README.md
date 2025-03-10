# Llama Summarizer

A Python utility for summarizing large text, markdown, PDF, and Word documents using Llama 3 running locally via Ollama.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)

## Features

- Summarizes text, markdown, PDF, and Word documents of any length
- Handles documents that exceed Llama 3's 8000 token context window
- Uses smart chunking with overlapping sections to ensure no information is lost
- Produces a comprehensive final summary that stays within the model's limits
- Automatically detects file formats based on extension

## Prerequisites

- Python 3.8+
- Ollama installed and running locally with Llama 3
- Run `ollama pull llama3:8b` to download the model

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/dynamikapps/llama-summarizer.git
   cd llama-summarizer
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following content:

   ```
   OLLAMA_URL=http://localhost:11434
   MODEL_NAME=llama3:8b
   ```

4. Make the shell scripts executable:
   ```
   chmod +x run_summarizer.sh test_all_formats.sh
   ```

## Quick Start

Summarize a document with default settings:

```bash
python summarize.py --input your_file.md
```

Or use the convenience script:

```bash
./run_summarizer.sh your_file.pdf
```

## Usage

```bash
python summarize.py --input your_file.md --output summary.md --max_tokens 1500 --chunk_size 3000 --overlap 600
```

### Supported File Formats

The script automatically detects the file type based on the extension:

- `.txt`, `.md`, `.markdown` - Treated as plain text/markdown
- `.pdf` - Extracts text from PDF files
- `.doc`, `.docx` - Extracts text from Word documents
- `.html`, `.htm` - Converts HTML to markdown

### Command Line Arguments

- `--input`: Path to the input file (text, markdown, PDF, or Word document)
- `--output`: Path for the output summary file (optional, defaults to `summary.md`)
- `--max_tokens`: Maximum tokens for the final summary (optional, defaults to 1000)
- `--chunk_size`: Number of tokens per chunk (optional, defaults to 4000)
- `--overlap`: Number of tokens to overlap between chunks (optional, defaults to 500)

### Running the Test Script

To test summarization across all supported file formats:

```bash
./test_all_formats.sh
```

This script:

1. Converts the sample document to HTML, DOCX, and PDF formats
2. Runs the summarizer on each format
3. Produces separate summary files for each format

Note: This requires `pandoc` and `wkhtmltopdf` to be installed for format conversion.

### Token Counting

While our script uses tiktoken to count tokens automatically, you can also check your document's token count using the OpenAI tokenizer:

- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)

This can be helpful to estimate how many chunks your document will be divided into before running the summarization process.

## How It Works

1. **Document Loading**: The input document is loaded and converted to text based on its file format.

2. **Text Chunking**: The document is split into overlapping chunks, each within the model's context window (default 4000 tokens with 500 token overlap).

3. **Chunk Summarization**: Each chunk is summarized individually by Llama 3.

4. **Recursive Processing**: If the combined summaries still exceed the context window, the process repeats with these summaries as input.

5. **Final Synthesis**: Once the combined summaries fit within the context window, a final coherent summary is generated.

6. **Output**: The final summary is saved to the specified output file.

## Customization

### Changing the Model

To use a different model, update the `MODEL_NAME` in your `.env` file:

```
MODEL_NAME=llama3:70b
```

### Adjusting Summarization Parameters

For longer or more detailed summaries, increase the `max_tokens` parameter:

```bash
python summarize.py --input your_file.md --max_tokens 2000
```

For documents with complex sections, increase the overlap between chunks:

```bash
python summarize.py --input your_file.md --overlap 800
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running with `ollama serve`

2. **Model Not Found**: Run `ollama pull llama3:8b` to download the model

3. **PDF Extraction Issues**: Some PDFs may have complex formatting or be scanned images, which can affect text extraction quality

4. **Long Processing Time**: Large documents with many chunks may take significant time to process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
