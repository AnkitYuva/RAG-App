"""
document_processor.py
---------------------
Handles all document-related operations:
  - Loading PDF files using PyPDFLoader
  - Splitting text into chunks using RecursiveCharacterTextSplitter
  - Returning chunks with metadata (source filename, page number)

CONCEPTS EXPLAINED:
  - Document Chunking: LLMs have a limited context window, so we cannot
    feed an entire document at once. We split documents into smaller
    overlapping chunks so relevant sections can be retrieved efficiently.
  - Chunk Overlap: A small overlap between consecutive chunks ensures
    that information at the boundary of a chunk is not lost.
"""

import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ─────────────────────────────────────────────
#  Default chunking parameters
#  These are easy to modify for experimentation
# ─────────────────────────────────────────────
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlapping characters between consecutive chunks


def load_and_split_pdfs(
    file_paths: List[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[Document], dict]:
    """
    Load one or more PDF files and split them into text chunks.

    Parameters
    ----------
    file_paths   : List of absolute paths to PDF files on disk.
    chunk_size   : Maximum number of characters in each chunk.
    chunk_overlap: Number of characters shared between consecutive chunks.

    Returns
    -------
    chunks : List of LangChain Document objects (each chunk with metadata).
    stats  : Dictionary with processing statistics for display in the UI.

    How it works step-by-step:
      1. PyPDFLoader reads each PDF page-by-page.
      2. RecursiveCharacterTextSplitter breaks the text at natural
         boundaries (paragraphs -> sentences -> words -> characters).
      3. Each chunk retains metadata: source filename and page number.
    """

    if not file_paths:
        raise ValueError("No files provided. Please upload at least one PDF.")

    all_documents: List[Document] = []
    pages_per_file: dict = {}

    # -- Step 1: Load each PDF ------------------------------------------------
    for path in file_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_name = os.path.basename(path)

        # Step 1: Load file based on extension (PDF, TXT, MD)
        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                pages = loader.load()          # returns one Document per page
            elif ext in [".txt", ".md", ".csv"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                pages = [Document(page_content=content, metadata={"source": file_name, "page": 0})]
            else:
                # Fallback text loader for other text formats
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                pages = [Document(page_content=content, metadata={"source": file_name, "page": 0})]
        except Exception as exc:
            raise RuntimeError(
                f"Could not load '{file_name}'. Error: {exc}"
            )

        if not pages or all(not p.page_content.strip() for p in pages):
            raise ValueError(f"'{file_name}' appears to be empty or unreadable.")

        # Attach clean filename to metadata
        for p in pages:
            p.metadata["source"] = file_name

        pages_per_file[file_name] = len(pages)
        all_documents.extend(pages)

    total_pages = len(all_documents)

    # -- Step 2: Split into chunks --------------------------------------------
    # RecursiveCharacterTextSplitter tries to split at "\n\n", then "\n",
    # then " ", then character-by-character - preserving meaning.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_documents)

    # Build statistics dictionary for the UI status panel
    stats = {
        "total_files"   : len(file_paths),
        "total_pages"   : total_pages,
        "total_chunks"  : len(chunks),
        "chunk_size"    : chunk_size,
        "chunk_overlap" : chunk_overlap,
        "pages_per_file": pages_per_file,
    }

    return chunks, stats


def format_processing_stats(stats: dict) -> str:
    """
    Format the processing statistics into a human-readable string
    for display in the Gradio status panel.
    """
    lines = [
        "✅ **Documents processed successfully!**\n",
        f"- 📄 **Files loaded:** {stats['total_files']}",
        f"- 📑 **Total pages:** {stats['total_pages']}",
        f"- 🔖 **Chunks created:** {stats['total_chunks']}",
        f"- ✂️  **Chunk size:** {stats['chunk_size']} chars",
        f"- 🔁 **Chunk overlap:** {stats['chunk_overlap']} chars",
        "",
        "**Per-file page counts:**",
    ]
    for fname, npages in stats["pages_per_file"].items():
        lines.append(f"  - `{fname}`: {npages} page(s)")

    return "\n".join(lines)
