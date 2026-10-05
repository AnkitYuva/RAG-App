"""
app.py
------
Main entry point for "Ask My Documents – RAG Assistant".

This file creates the Gradio web interface and connects:
  - document_processor.py  (PDF loading + chunking)
  - rag_engine.py          (embeddings + FAISS + LLM)

Run with:
    python app.py
Then open http://127.0.0.1:7860 in your browser.
"""

import os
import traceback
from typing import List, Optional

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

import document_processor as dp
import rag_engine as rag


def get_default_api_key() -> str:
    """Get any pre-configured API key from environment."""
    return (
        os.getenv("NVIDIA_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
    )


def process_documents(uploaded_files, chunk_size: int = 800, chunk_overlap: int = 150, api_key: str = "") -> tuple:
    """
    Called when the user clicks 'Process Documents'.
    Loads PDFs, splits into chunks, and builds FAISS vector store.
    """
    if not uploaded_files:
        return (
            "⚠️ **No files selected.** Please upload at least one PDF file or click 'Load Sample Document'.",
            False,
            gr.update(interactive=False),
        )

    file_paths = []
    for f in uploaded_files:
        if hasattr(f, "name"):
            file_paths.append(f.name)
        elif isinstance(f, str):
            file_paths.append(f)
        elif hasattr(f, "path"):
            file_paths.append(f.path)

    try:
        rag.reset()

        # Step 1: Document loading & chunking
        chunks, stats = dp.load_and_split_pdfs(
            file_paths=file_paths,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )

        # Step 2: Vector embedding & FAISS store creation
        rag.build_vector_store(chunks, api_key=api_key.strip() if api_key else None)

        # Step 3: Success stats display
        status_text = dp.format_processing_stats(stats)
        return (
            status_text,
            True,
            gr.update(interactive=True),
        )

    except FileNotFoundError as e:
        return (f"❌ **File Error:** {e}", False, gr.update(interactive=False))
    except ValueError as e:
        return (f"❌ **Validation Error:** {e}", False, gr.update(interactive=False))
    except RuntimeError as e:
        return (f"❌ **Processing Error:** {e}", False, gr.update(interactive=False))
    except Exception as e:
        tb = traceback.format_exc()
        return (
            f"❌ **Unexpected Error:**\n```\n{tb}\n```",
            False,
            gr.update(interactive=False),
        )


def load_sample_document() -> tuple:
    """Loads the pre-packaged sample syllabus PDF for instant 1-click testing."""
    sample_path = os.path.abspath("data/sample_documents/AI_Course_Syllabus.pdf")
    if not os.path.exists(sample_path):
        return (
            None,
            "⚠️ Sample file not found. Generating sample documents...",
            False,
            gr.update(interactive=False),
        )

    try:
        rag.reset()
        chunks, stats = dp.load_and_split_pdfs([sample_path])
        rag.build_vector_store(chunks)
        status_text = dp.format_processing_stats(stats)
        status_text = "🎉 **Sample Document Loaded:** `AI_Course_Syllabus.pdf`\n\n" + status_text
        return (
            [sample_path],
            status_text,
            True,
            gr.update(interactive=True),
        )
    except Exception as e:
        return (
            None,
            f"❌ **Error loading sample document:** {e}",
            False,
            gr.update(interactive=False),
        )


def ask_question(
    question: str,
    docs_ready: bool,
    provider: str,
    api_key: str,
    prompt_strategy: str,
    custom_instructions: str,
    top_k: int,
) -> tuple:
    """
    Called when the user submits a question.
    Runs similarity retrieval, injects context into the selected prompt strategy,
    and generates LLM answer.
    """
    if not docs_ready or not rag.is_ready():
        return (
            "⚠️ **Documents not processed.**\n\n"
            "Please upload document files and click **'Process Documents'** first.",
            "",
            "No prompt assembled yet.",
        )

    if not question or not question.strip():
        return (
            "⚠️ **Empty question.** Please enter a question to search your documents.",
            "",
            "No prompt assembled yet.",
        )

    try:
        answer, sources, assembled_prompt = rag.answer_question(
            question=question,
            api_key=api_key.strip() if api_key else None,
            provider=provider,
            strategy=prompt_strategy,
            custom_instructions=custom_instructions,
            top_k=int(top_k),
        )
        sources_text = rag.format_sources(sources)
        answer_text = f"### 💡 Answer ({prompt_strategy})\n\n{answer}"
        return answer_text, sources_text, assembled_prompt

    except ValueError as e:
        return f"⚠️ **Input Error:** {e}", "", ""
    except RuntimeError as e:
        return f"❌ **Configuration Notice:**\n\n{e}", "", ""
    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ **Error generating response:**\n```\n{tb}\n```", "", ""


def clear_all() -> tuple:
    """Reset the application state."""
    rag.reset()
    return (
        None,                                 # file upload
        "Upload documents above or click **'Load Sample Document'** to begin.",
        "",                                   # question box
        "",                                   # answer box
        "",                                   # sources box
        "",                                   # prompt inspector
        False,                                # docs_ready
        gr.update(interactive=False),         # ask_btn
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STYLING & LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_DIAGRAM = """\
[User Document: PDF]
         │
         ▼
[PyPDFLoader: Page Extraction]
         │
         ▼
[RecursiveCharacterTextSplitter: Chunks + Overlap]
         │
         ▼
[Embedding Model: Vector Numerical Representation]
         │
         ▼
[FAISS Vector Store: Fast Indexing]
         │
  [User Question] ──▶ [Embed Query] ──▶ [Similarity Search (Top-K)]
                                                     │
                                                     ▼
                                            [Context Assembly]
                                                     │
                                                     ▼
                                            [Strict Prompting]
                                                     │
                                                     ▼
                                            [LLM Grounded Answer + Citations]
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Ask My Documents - RAG Assistant") as demo:
        docs_ready = gr.State(False)

        # ── Header & Styles ────────────────────────────────────────────────
        gr.Markdown(
            """
# Ask My Documents - RAG Assistant
Upload PDF, text, or Markdown files. Ask questions and get answers grounded in the uploaded documents, with source references.
"""
        )

        # ── Pipeline Overview ──────────────────────────────────────────────
        with gr.Accordion("ℹ️ System Architecture & Workflow", open=False):
            gr.Markdown("""
### Complete RAG Workflow Explained:
1. **Document Loading (`PyPDFLoader` / Text Loaders)**: Extracts raw text and preserves metadata (source filename and page numbers).
2. **Text Chunking (`RecursiveCharacterTextSplitter`)**: Splits text at natural paragraph and sentence boundaries with chunk overlap to preserve context across boundaries.
3. **Embeddings & Vector Store (`FAISS`)**: Transforms chunks into dense vectors and indexes them for high-speed nearest-neighbor search.
4. **Semantic Retrieval**: Performs similarity search with score to find the Top-K chunks closest in semantic meaning to your query.
5. **Prompt Design**: Injects retrieved chunks into prompt strategies such as Strict Grounding, Evidence-First, Few-Shot, Executive Briefing, or Custom Persona.
6. **Grounded Generation (`LLM`)**: The model answers exclusively from the context with verifiable citations.
""")
            gr.Markdown(f"```text\n{PIPELINE_DIAGRAM}\n```")

        # ── Settings & Prompt Design ───────────────────────────────────────
        with gr.Accordion("⚙️ Model, API & Chunking Configuration", open=False):
            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    choices=["Auto-detect", "Google Gemini"],
                    value="Auto-detect",
                    label="LLM Provider",
                    info="Use Auto-detect or Google Gemini with GOOGLE_API_KEY.",
                )
                api_key_input = gr.Textbox(
                    label="API Key (optional if configured in .env)",
                    placeholder="Enter your API key or leave blank to use .env",
                    type="password",
                    value=get_default_api_key(),
                )

            with gr.Row():
                chunk_size_slider = gr.Slider(
                    minimum=200,
                    maximum=2000,
                    step=50,
                    value=800,
                    label="Chunk Size (Characters)",
                    info="Maximum length of each text chunk",
                )
                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=400,
                    step=25,
                    value=150,
                    label="Chunk Overlap (Characters)",
                    info="Overlap between adjacent chunks to maintain context",
                )

        # ══ PROMPT DESIGN CONTROLS ══════════════════════════════════════════
        with gr.Accordion("🎯 Prompt Design & Retrieval Controls", open=True):
            gr.Markdown("""
Select from **industry-standard prompt engineering patterns** designed to eliminate hallucinations, enforce structured outputs, or provide reasoning traces.
""")
            with gr.Row():
                prompt_strategy_dropdown = gr.Dropdown(
                    choices=list(rag.PROMPT_STRATEGIES.keys()),
                    value="Strict Grounding (Anti-Hallucination)",
                    label="Prompt Engineering Strategy",
                    info="Controls system persona, reasoning instructions, and output constraints",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                    label="Top-K Retrieved Chunks",
                    info="Number of relevant chunks passed as context",
                )

            custom_instructions_box = gr.Textbox(
                label="Custom Persona / System Instructions (Active when 'Custom Persona / Prompt' is selected)",
                placeholder="e.g., You are an Academic Auditor. Provide a strict analysis with bullet points and page citations.",
                value="You are an expert AI Academic Advisor. Answer the question thoroughly and concisely using only the document context provided.",
                lines=2,
            )

        gr.Markdown("---")

        # ══ STEP 1: Upload Documents ════════════════════════════════════════
        gr.Markdown("### 📤 Upload & Process Documents")
        with gr.Row():
            with gr.Column(scale=3):
                file_upload = gr.File(
                    label="Upload PDF, TXT, or Markdown Documents",
                    file_types=[".pdf", ".txt", ".md"],
                    file_count="multiple",
                    height=150,
                )
            with gr.Column(scale=1):
                process_btn = gr.Button("Process Documents", variant="primary", size="lg")
                sample_btn = gr.Button("Load Sample Document", variant="secondary", size="sm")
                clear_btn = gr.Button("Clear / Reset", variant="secondary", size="sm")

        status_box = gr.Markdown(
            value="Upload documents above or click **'Load Sample Document'** to get started.",
        )

        gr.Markdown("---")

        # ══ STEP 2: Ask Question ════════════════════════════════════════════
        gr.Markdown("### 💬 Ask a Question")
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are the prerequisites and grading breakdown for the course?",
            lines=2,
            max_lines=4,
        )
        ask_btn = gr.Button("Ask Question", variant="primary", size="lg", interactive=False)

        gr.Markdown("---")

        # ══ STEP 3: Answers & Citations ═════════════════════════════════════
        gr.Markdown("### 📋 Answer & Sources")
        answer_box = gr.Markdown(value="")

        with gr.Accordion("📎 Retrieved Source Documents, Chunks & Similarity Scores", open=True):
            sources_box = gr.Markdown(value="")

        with gr.Accordion("🔍 Inspect Live Assembled Prompt", open=False):
            gr.Markdown("_This panel shows the exact prompt assembled by the prompt engine and sent to the LLM:_")
            prompt_inspector_box = gr.Textbox(
                label="Full Assembled Prompt Sent to Model",
                lines=10,
                max_lines=25,
                interactive=False,
            )

        # ── Sample Questions ───────────────────────────────────────────────
        with gr.Accordion("💡 Suggested Questions", open=True):
            gr.Markdown("""
Try these after loading the sample document:
1. *"What are the prerequisites for this course?"*
2. *"What is the grading weightage for the Capstone Project and Midterm?"*
3. *"What are the penalties for late project submissions?"*
4. *"What is the price of the recommended textbooks?"*
5. *"What competencies will students develop?"*
""")

        # ── Event Wiring ───────────────────────────────────────────────────
        process_btn.click(
            fn=process_documents,
            inputs=[file_upload, chunk_size_slider, chunk_overlap_slider, api_key_input],
            outputs=[status_box, docs_ready, ask_btn],
        )

        sample_btn.click(
            fn=load_sample_document,
            inputs=[],
            outputs=[file_upload, status_box, docs_ready, ask_btn],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[
                question_box,
                docs_ready,
                provider_dropdown,
                api_key_input,
                prompt_strategy_dropdown,
                custom_instructions_box,
                top_k_slider,
            ],
            outputs=[answer_box, sources_box, prompt_inspector_box],
        )

        question_box.submit(
            fn=ask_question,
            inputs=[
                question_box,
                docs_ready,
                provider_dropdown,
                api_key_input,
                prompt_strategy_dropdown,
                custom_instructions_box,
                top_k_slider,
            ],
            outputs=[answer_box, sources_box, prompt_inspector_box],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                file_upload,
                status_box,
                question_box,
                answer_box,
                sources_box,
                prompt_inspector_box,
                docs_ready,
                ask_btn,
            ],
        )

    return demo


if __name__ == "__main__":
    os.makedirs("data/sample_documents", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    server_port = int(os.getenv("PORT", 7860))
    server_host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    app = build_app()
    app.launch(
        server_name=server_host,
        server_port=server_port,
        share=False,
        inbrowser=False,
    )
