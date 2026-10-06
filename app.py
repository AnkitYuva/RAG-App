"""
Minimal Colab-friendly Gradio RAG app.

Run locally:
    python app.py

Run in Colab:
    !pip install -r requirements.txt
    !python app.py
"""

import hashlib
import math
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

try:
    import spaces

    zero_gpu = spaces.GPU if os.getenv("SPACE_ID") else (lambda fn: fn)
except Exception:

    def zero_gpu(fn):
        return fn


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_DIMS = 4096


def get_default_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "")


def read_file_text(path: str) -> List[Dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    docs = []

    if suffix == ".pdf":
        reader = PdfReader(path)
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    {
                        "text": text,
                        "source": file_path.name,
                        "page": page_index + 1,
                    }
                )
    elif suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append({"text": text, "source": file_path.name, "page": 1})
    else:
        raise ValueError(f"Unsupported file type: {file_path.name}")

    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def embed_text(text: str) -> List[float]:
    vector = [0.0] * EMBED_DIMS
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) > 2]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % EMBED_DIMS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def get_file_paths(uploaded_files) -> List[str]:
    paths = []
    for item in uploaded_files or []:
        if isinstance(item, str):
            paths.append(item)
        elif hasattr(item, "name"):
            paths.append(item.name)
        elif hasattr(item, "path"):
            paths.append(item.path)
    return paths


@zero_gpu
def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    if not uploaded_files:
        return "Please upload at least one PDF, TXT, or Markdown file.", None, gr.update(interactive=False)

    try:
        paths = get_file_paths(uploaded_files)
        index = []
        total_pages = 0

        for path in paths:
            pages = read_file_text(path)
            total_pages += len(pages)

            for page in pages:
                for chunk in chunk_text(page["text"], int(chunk_size), int(chunk_overlap)):
                    index.append(
                        {
                            "text": chunk,
                            "source": page["source"],
                            "page": page["page"],
                            "embedding": embed_text(chunk),
                        }
                    )

        if not index:
            return "No readable text was found in the uploaded documents.", None, gr.update(interactive=False)

        status = (
            "Documents processed successfully.\n\n"
            f"- Files: {len(paths)}\n"
            f"- Pages/text files: {total_pages}\n"
            f"- Chunks: {len(index)}"
        )
        return status, index, gr.update(interactive=True)

    except Exception:
        return f"Processing failed:\n\n```\n{traceback.format_exc()}\n```", None, gr.update(interactive=False)


@zero_gpu
def load_sample_document():
    sample_path = os.path.abspath("data/sample_documents/AI_Course_Syllabus.pdf")
    if not os.path.exists(sample_path):
        return None, "Sample document not found.", None, gr.update(interactive=False)

    status, index, ask_update = process_documents([sample_path], CHUNK_SIZE, CHUNK_OVERLAP)
    return [sample_path], status, index, ask_update


def retrieve(question: str, index: List[Dict], top_k: int) -> List[Dict]:
    query_embedding = embed_text(question)
    scored = []

    for item in index:
        scored.append((cosine_similarity(query_embedding, item["embedding"]), item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, item in scored[:top_k]:
        result = dict(item)
        result["score"] = score
        results.append(result)
    return results


def build_prompt(question: str, sources: List[Dict]) -> str:
    context_blocks = []
    for i, source in enumerate(sources, start=1):
        context_blocks.append(
            f"[Source {i} | File: {source['source']} | Page: {source['page']}]\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_blocks)
    return f"""You are a document question-answering assistant.

Answer the question using ONLY the document context below.
If the answer is not present, say: "I could not find this information in the uploaded documents."
Use citations like [Source 1, Page 2].

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def call_gemini(prompt: str, api_key: str) -> str:
    key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Add it in the API key box, Colab secrets, .env, "
            "or Hugging Face Space secrets."
        )

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def offline_answer(question: str, sources: List[Dict], error_text: str) -> str:
    """Fallback answer for demos when the Gemini key is missing or invalid."""
    if "API_KEY_INVALID" in error_text or "API key not valid" in error_text:
        reason = "The Google API key is invalid."
    elif "Missing GOOGLE_API_KEY" in error_text:
        reason = "No Google API key was provided."
    else:
        reason = "Gemini could not generate a response."

    if not sources:
        return f"{reason}\n\nNo document sources were retrieved."

    best = sources[0]
    snippet = best["text"][:900]
    if len(best["text"]) > 900:
        snippet += "..."

    return (
        f"{reason}\n\n"
        "**Offline demo answer from retrieved context:**\n\n"
        f"The most relevant uploaded document content is from `{best['source']}`, "
        f"page {best['page']} [Source 1, Page {best['page']}].\n\n"
        f"> {snippet}\n\n"
        "For full AI-generated answering, paste a valid Google Gemini API key in "
        "Settings or set `GOOGLE_API_KEY` before starting the app."
    )


def format_sources(sources: List[Dict]) -> str:
    if not sources:
        return "No sources retrieved."

    lines = []
    for i, source in enumerate(sources, start=1):
        snippet = source["text"][:500]
        if len(source["text"]) > 500:
            snippet += "..."
        lines.append(
            f"**Source {i}**\n"
            f"- File: `{source['source']}`\n"
            f"- Page: {source['page']}\n"
            f"- Similarity: `{source['score']:.3f}`\n\n"
            f"> {snippet}"
        )
    return "\n\n---\n\n".join(lines)


@zero_gpu
def ask_question(question: str, index, api_key: str, top_k: int):
    if not index:
        return "Please upload and process documents first.", "", ""

    if not question or not question.strip():
        return "Please enter a question.", "", ""

    try:
        sources = retrieve(question.strip(), index, int(top_k))
        prompt = build_prompt(question.strip(), sources)
        try:
            answer = call_gemini(prompt, api_key)
        except Exception as exc:
            answer = offline_answer(question.strip(), sources, str(exc))
        return answer, format_sources(sources), prompt
    except Exception:
        return f"Retrieval failed:\n\n```\n{traceback.format_exc()}\n```", "", ""


def clear_all():
    return None, "Upload documents to begin.", None, "", "", "", gr.update(interactive=False)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Ask My Documents - RAG Assistant") as demo:
        rag_index = gr.State(None)

        gr.Markdown(
            """
# Ask My Documents - RAG Assistant
Upload documents, process them, ask questions, and get grounded answers with sources.
"""
        )

        with gr.Accordion("Settings", open=False):
            api_key = gr.Textbox(
                label="Google API Key",
                type="password",
                value=get_default_api_key(),
                placeholder="Leave blank if GOOGLE_API_KEY is already configured",
            )
            with gr.Row():
                chunk_size = gr.Slider(300, 1500, value=CHUNK_SIZE, step=50, label="Chunk Size")
                chunk_overlap = gr.Slider(0, 300, value=CHUNK_OVERLAP, step=25, label="Chunk Overlap")
                top_k = gr.Slider(1, 6, value=TOP_K, step=1, label="Top-K Sources")

        file_upload = gr.File(
            label="Upload PDF, TXT, or Markdown Documents",
            file_types=[".pdf", ".txt", ".md"],
            file_count="multiple",
        )

        with gr.Row():
            process_btn = gr.Button("Process Documents", variant="primary")
            sample_btn = gr.Button("Load Sample Document")
            clear_btn = gr.Button("Clear")

        status = gr.Markdown("Upload documents to begin.")

        question = gr.Textbox(label="Ask a Question", lines=2, placeholder="What is this document about?")
        ask_btn = gr.Button("Ask", variant="primary", interactive=False)

        answer = gr.Markdown(label="Answer")
        with gr.Accordion("Sources", open=True):
            sources = gr.Markdown()
        with gr.Accordion("Prompt", open=False):
            prompt = gr.Textbox(lines=10, interactive=False)

        process_btn.click(
            process_documents,
            inputs=[file_upload, chunk_size, chunk_overlap],
            outputs=[status, rag_index, ask_btn],
        )
        sample_btn.click(
            load_sample_document,
            inputs=[],
            outputs=[file_upload, status, rag_index, ask_btn],
        )
        ask_btn.click(
            ask_question,
            inputs=[question, rag_index, api_key, top_k],
            outputs=[answer, sources, prompt],
        )
        question.submit(
            ask_question,
            inputs=[question, rag_index, api_key, top_k],
            outputs=[answer, sources, prompt],
        )
        clear_btn.click(
            clear_all,
            inputs=[],
            outputs=[file_upload, status, rag_index, question, answer, sources, prompt, ask_btn],
        )

    return demo


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    running_in_colab = bool(os.getenv("COLAB_RELEASE_TAG"))

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=running_in_colab,
        inbrowser=False,
    )
