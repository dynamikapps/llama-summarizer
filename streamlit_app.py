import streamlit as st
import os
import tempfile
import time
import requests
import tiktoken
import PyPDF2
import docx
import io
from pathlib import Path
from markdownify import markdownify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants from environment
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:8b")

# Initialize tokenizer
# Using this as a proxy for Llama 3 tokenization
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Page configuration
st.set_page_config(
    page_title="Llama Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div {
        background-image: linear-gradient(to right, #4CAF50, #8BC34A);
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .summary-container {
        border-left: 3px solid #4CAF50;
        padding-left: 20px;
        margin: 20px 0;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions from summarize.py


def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(ENCODING.encode(text))


def extract_text_from_pdf_bytes(file_bytes):
    """Extract text content from PDF file bytes."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n\n"
    return text


def extract_text_from_docx_bytes(file_bytes):
    """Extract text content from a Word document bytes."""
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)


def process_uploaded_file(uploaded_file):
    """Process the uploaded file and return its text content."""
    if uploaded_file is None:
        return None

    # Get the file extension
    file_extension = Path(uploaded_file.name).suffix.lower()

    # Read the file content
    file_bytes = uploaded_file.getvalue()

    # Handle different file types
    if file_extension == '.pdf':
        return extract_text_from_pdf_bytes(file_bytes)

    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx_bytes(file_bytes)

    elif file_extension in ['.html', '.htm']:
        content = file_bytes.decode('utf-8')
        return markdownify(content)

    else:  # Text, Markdown, or other plain text formats
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                st.error(f"Failed to decode file: {e}")
                return None


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
        st.error(f"Error querying Ollama API: {e}")
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


def recursive_summarize(text, chunk_size=4000, overlap=500, max_tokens=1000, progress_bar=None):
    """Recursively summarize a text document that may exceed the model's context window."""
    chunks = chunk_text(text, chunk_size, overlap)

    # If we only have one chunk, just summarize it directly
    if len(chunks) == 1:
        if progress_bar:
            progress_bar.progress(0.5)
            progress_bar.text("Summarizing single chunk...")
        result = summarize_chunk(chunks[0], max_tokens)
        if progress_bar:
            progress_bar.progress(1.0)
        return result

    st.write(f"Document divided into {len(chunks)} chunks for processing...")

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        if progress_bar:
            # Update progress bar to show progression through chunks
            progress_bar.progress((i) / (len(chunks) + 1))
            progress_bar.text(f"Summarizing chunk {i+1} of {len(chunks)}...")

        chunk_summary = summarize_chunk(chunk, max_tokens=max_tokens)
        summaries.append(chunk_summary)

    # Check if the combined summaries exceed the context window
    combined_summary_text = "\n\n".join(summaries)
    combined_tokens = count_tokens(combined_summary_text)

    # If combined summaries are still too large, recursively summarize them
    if combined_tokens > chunk_size:
        if progress_bar:
            progress_bar.text(
                f"Combined summaries ({combined_tokens} tokens) exceed the context window. Performing recursive summarization...")

        return recursive_summarize(combined_summary_text, chunk_size, overlap, max_tokens, progress_bar)

    # Otherwise, combine the summaries
    if progress_bar:
        progress_bar.progress((len(chunks)) / (len(chunks) + 1))
        progress_bar.text("Generating final summary...")

    final_summary = combine_summaries(summaries, max_tokens)

    if progress_bar:
        progress_bar.progress(1.0)
        progress_bar.text("Summary completed!")

    return final_summary


# Streamlit UI
st.title("🦙 Llama Document Summarizer")
st.markdown("""
This app uses Llama 3 running locally via Ollama to summarize documents of any length.
Upload your document, adjust the summarization parameters if needed, and get a concise summary!
""")

# Check if Ollama is available


@st.cache_data(ttl=300)  # Cache the result for 5 minutes
def check_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = [model['name']
                      for model in response.json().get('models', [])]
            if MODEL_NAME in models:
                return True, f"Connected to Ollama. Using model: {MODEL_NAME}"
            else:
                return False, f"Model {MODEL_NAME} not found. Please run: ollama pull {MODEL_NAME}"
        return False, "Ollama connection error. Please make sure Ollama is running."
    except:
        return False, "Failed to connect to Ollama. Please make sure Ollama is running at the URL specified in .env"


ollama_ok, ollama_message = check_ollama_connection()

if not ollama_ok:
    st.warning(ollama_message)
else:
    st.success(ollama_message)

# Sidebar for parameters
st.sidebar.header("Summarization Parameters")

max_tokens = st.sidebar.slider(
    "Maximum tokens in final summary",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100,
    help="Controls the length of the final summary."
)

chunk_size = st.sidebar.slider(
    "Chunk size (tokens)",
    min_value=1000,
    max_value=7000,
    value=4000,
    step=500,
    help="Size of each text chunk sent to the model. Max 8000 for Llama 3."
)

overlap = st.sidebar.slider(
    "Chunk overlap (tokens)",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
    help="Overlap between chunks to maintain context continuity."
)

# File upload
st.header("1. Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['txt', 'md', 'markdown', 'html', 'htm', 'pdf', 'doc', 'docx'],
    help="Supported formats: Markdown, Text, HTML, PDF, and Word documents."
)

# Process the uploaded file
if uploaded_file is not None:
    with st.expander("Document Details", expanded=True):
        file_name = uploaded_file.name
        file_type = uploaded_file.type
        file_size = uploaded_file.size / 1024  # Convert to KB

        col1, col2, col3 = st.columns(3)
        col1.metric("File Name", file_name)
        col2.metric("File Type", file_type if file_type else Path(
            file_name).suffix[1:].upper())
        col3.metric("File Size", f"{file_size:.2f} KB")

        # Process the file
        with st.spinner("Processing file..."):
            text_content = process_uploaded_file(uploaded_file)

            if text_content:
                token_count = count_tokens(text_content)
                st.metric("Token Count", token_count)

                if token_count > chunk_size:
                    st.info(
                        f"This document will be processed in approximately {token_count // (chunk_size - overlap) + 1} chunks.")
            else:
                st.error("Failed to extract text from the file.")

    # Summarize button
    if text_content and st.button("Generate Summary", type="primary", disabled=not ollama_ok):
        st.header("2. Summarization Results")

        # Create a progress bar and status text
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()

        # Set initial status
        status_text.text("Starting summarization process...")

        # Create a wrapper for the progress bar and status text
        class ProgressWrapper:
            def __init__(self, bar, status_text):
                self.bar = bar
                self.status_text = status_text

            def progress(self, value):
                self.bar.progress(value)

            def text(self, message):
                self.status_text.text(message)

        progress_wrapper = ProgressWrapper(progress_bar, status_text)

        # Run the summarization
        try:
            start_time = time.time()
            summary = recursive_summarize(
                text_content,
                chunk_size=chunk_size,
                overlap=overlap,
                max_tokens=max_tokens,
                progress_bar=progress_wrapper
            )
            end_time = time.time()

            if summary:
                # Display the summary
                st.markdown("### Summary")
                st.markdown('<div class="summary-container">',
                            unsafe_allow_html=True)
                st.markdown(summary)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display metadata
                st.success(
                    f"Summary generated in {end_time - start_time:.2f} seconds")

                # Word and token counts
                summary_word_count = len(summary.split())
                summary_token_count = count_tokens(summary)

                col1, col2, col3 = st.columns(3)
                col1.metric("Original Tokens", token_count)
                col2.metric("Summary Tokens", summary_token_count)
                col3.metric("Compression Ratio",
                            f"{token_count / summary_token_count:.1f}x")

                # Download button
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"{Path(file_name).stem}_summary.md",
                    mime="text/markdown"
                )
            else:
                st.error(
                    "Failed to generate summary. Check the Ollama connection and try again.")
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")

# Footer
st.sidebar.markdown("""---
## About
This app uses [Llama 3](https://ai.meta.com/llama/) via [Ollama](https://ollama.ai/) to provide document summarization.

## Resources
- [GitHub Repository](https://github.com/yourusername/llama_summarizer)
- [Report an Issue](https://github.com/yourusername/llama_summarizer/issues)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
""")

# Run the app with: streamlit run streamlit_app.py
