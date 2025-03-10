#!/usr/bin/env python3
import argparse
import os
import json
import requests
import tiktoken
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from markdownify import markdownify
import PyPDF2
import docx
import io

# Load environment variables
load_dotenv()

# Constants from environment
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:8b")

# Initialize tokenizer
# Using this as a proxy for Llama 3 tokenization
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")


def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(ENCODING.encode(text))


def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    return text


def extract_text_from_docx(file_path):
    """Extract text content from a Word document."""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def load_document(file_path):
    """Load a document from a file path, converting to plain text if necessary."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Handle PDF files
    if suffix == '.pdf':
        content = extract_text_from_pdf(file_path)
        return content

    # Handle Word documents
    elif suffix in ['.docx', '.doc']:
        content = extract_text_from_docx(file_path)
        return content

    # Handle HTML files
    elif suffix in ['.html', '.htm']:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        content = markdownify(content)
        return content

    # Handle text and markdown files
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try different encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                return content
            except Exception as e:
                raise Exception(f"Failed to open file {file_path}: {e}")


def chunk_text(text, chunk_size=4000, overlap=500):
    """Split text into overlapping chunks of approximately chunk_size tokens."""
    tokens = ENCODING.encode(text)
    total_tokens = len(tokens)
    chunks = []

    # If text is smaller than chunk_size, return it as a single chunk
    if total_tokens <= chunk_size:
        return [text]

    # Create chunks with overlap
    for i in range(0, total_tokens, chunk_size - overlap):
        chunk_end = min(i + chunk_size, total_tokens)
        if chunk_end == total_tokens:  # Last chunk
            chunk_tokens = tokens[i:chunk_end]
        else:
            chunk_tokens = tokens[i:chunk_end]

        chunk_text = ENCODING.decode(chunk_tokens)
        chunks.append(chunk_text)

        if chunk_end == total_tokens:
            break

    return chunks


def query_ollama(prompt, model=MODEL_NAME, system=None, max_tokens=None):
    """Send a query to Ollama API and return the response."""
    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    if system:
        payload["system"] = system

    if max_tokens:
        payload["options"] = {"num_predict": max_tokens}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama API: {e}")
        return None


def summarize_chunk(chunk, max_tokens=1000):
    """Summarize a single chunk of text."""
    system_prompt = "You are an expert summarizer. Your task is to create a concise, comprehensive summary of the provided text. Focus on the main ideas, key points, and significant details. Preserve the original meaning and critical information."

    prompt = f"Please summarize the following text in approximately {max_tokens} tokens:\n\n{chunk}"

    return query_ollama(prompt, system=system_prompt, max_tokens=max_tokens)


def combine_summaries(summaries, max_tokens=1000):
    """Combine multiple summaries into a single, coherent summary."""
    combined_text = "\n\n".join(
        [f"Section {i+1}:\n{summary}" for i, summary in enumerate(summaries)])

    system_prompt = "You are an expert at synthesizing information. Your task is to create a single coherent summary that combines the key points from all provided section summaries. Organize the information logically, avoid repetition, and ensure the final summary is comprehensive yet concise."

    prompt = f"The following are summaries of different sections of a larger document. Please create a single coherent summary that integrates all the important information from these section summaries. Your summary should be approximately {max_tokens} tokens.\n\n{combined_text}"

    return query_ollama(prompt, system=system_prompt, max_tokens=max_tokens)


def recursive_summarize(text, chunk_size=4000, overlap=500, max_tokens=1000):
    """Recursively summarize a text document that may exceed the model's context window."""
    chunks = chunk_text(text, chunk_size, overlap)

    # If we only have one chunk, just summarize it directly
    if len(chunks) == 1:
        return summarize_chunk(chunks[0], max_tokens)

    print(f"Document divided into {len(chunks)} chunks for processing...")

    # Summarize each chunk in parallel
    summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        chunk_summary = summarize_chunk(chunk, max_tokens=max_tokens)
        summaries.append(chunk_summary)

    # Check if the combined summaries exceed the context window
    combined_summary_text = "\n\n".join(summaries)
    combined_tokens = count_tokens(combined_summary_text)

    # If combined summaries are still too large, recursively summarize them
    if combined_tokens > chunk_size:
        print(
            f"Combined summaries ({combined_tokens} tokens) exceed the context window. Performing recursive summarization...")
        return recursive_summarize(combined_summary_text, chunk_size, overlap, max_tokens)

    # Otherwise, combine the summaries
    print("Generating final summary...")
    return combine_summaries(summaries, max_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize large documents using Llama 3 via Ollama")
    parser.add_argument("--input", required=True,
                        help="Path to the input file")
    parser.add_argument("--output", default="summary.md",
                        help="Path to the output summary file")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens for the final summary")
    parser.add_argument("--chunk_size", type=int, default=4000,
                        help="Number of tokens per chunk")
    parser.add_argument("--overlap", type=int, default=500,
                        help="Number of tokens to overlap between chunks")

    args = parser.parse_args()

    print(f"Loading document: {args.input}")
    document = load_document(args.input)

    total_tokens = count_tokens(document)
    print(f"Document loaded: {total_tokens} tokens")

    # Check if the model is available
    try:
        test_response = requests.get(f"{OLLAMA_URL}/api/tags")
        test_response.raise_for_status()
        models = [model['name'] for model in test_response.json()['models']]
        if MODEL_NAME not in models:
            print(
                f"Warning: {MODEL_NAME} not found in available models. You may need to run 'ollama pull {MODEL_NAME}'")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        print(f"Make sure Ollama is running at {OLLAMA_URL}")
        return

    print(f"Beginning summarization process...")
    summary = recursive_summarize(
        document,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_tokens=args.max_tokens
    )

    # Save the summary to the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"Summary saved to {args.output}")
    print(f"Summary length: {count_tokens(summary)} tokens")


if __name__ == "__main__":
    main()
