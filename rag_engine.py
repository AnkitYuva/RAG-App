"""
rag_engine.py
-------------
Core RAG (Retrieval-Augmented Generation) engine.

Responsibilities:
  1. Build the FAISS vector store from document chunks.
  2. Expose a retriever that performs similarity search.
  3. Run the full RAG pipeline: question -> embedding -> retrieval -> LLM -> answer + sources.

CONCEPTS EXPLAINED:
  - Embeddings     : Dense numerical vectors that capture semantic meaning.
  - FAISS          : Facebook AI Similarity Search – fast nearest-neighbour vector search.
  - Top-K Retrieval: Retrieves the k most relevant chunks (default k=4) as context.
  - Strict Prompt  : Instructs LLM to answer ONLY from context, avoiding hallucinations.
"""

import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# Safe imports compatible with all LangChain versions
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Number of relevant chunks retrieved per question
TOP_K = 4

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2: PROFESSIONAL PROMPT DESIGN
# ─────────────────────────────────────────────────────────────────────────────
# 1. Strict Grounded System Prompt (Anti-Hallucination & Defensiveness)
STRICT_RAG_PROMPT = """You are a highly precise Document Verification and Question-Answering Assistant.

OBJECTIVE:
Answer the user's question using EXCLUSIVELY the provided document context below. 

STRICT OPERATIONAL RULES:
1. Grounding: Rely ONLY on the explicitly stated facts in the DOCUMENT CONTEXT. Do NOT assume, extrapolate, or bring in outside knowledge.
2. Defensiveness & Fallback: If the exact information needed to answer the question is not present in the context, you MUST respond verbatim with:
   "I could not find this information in the uploaded documents."
3. Citations: Whenever you cite a fact, reference the source in brackets, e.g., [Source 1, Page 2].
4. Objectivity: Maintain an objective, professional, and concise tone.

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION: {question}

GROUNDED ANSWER:"""

# 2. Evidence-first prompting (shows retrieved facts without exposing hidden reasoning)
EVIDENCE_FIRST_RAG_PROMPT = """You are an analytical Document QA assistant performing grounded evidence review.

OBJECTIVE:
Use the document context to produce an accurate, verified answer with visible evidence.

EVIDENCE WORKFLOW:
1. Extract only facts from the context that directly support the answer.
2. Verify whether those facts are sufficient to answer the question.
3. Avoid outside facts, assumptions, and hallucinations.
4. Formulate a clear, direct answer with explicit source and page citations.

If the context contains insufficient information, clearly state what is missing and say:
"I could not find this information in the uploaded documents."

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION: {question}

Follow this exact output format:
**Evidence Used:**
- <Relevant facts with source and page citations>

**Final Answer:**
<Synthesized response with [Source X, Page Y] citations>"""

# 3. Few-Shot In-Context Exemplar Prompting
FEW_SHOT_RAG_PROMPT = """You are a professional enterprise QA assistant. Follow the exact answering behavior shown in the examples below.

### EXAMPLE 1 (Answer directly present in context):
Context:
[Source 1 | File: Handbook.pdf | Page: 4]
Undergraduate students may borrow up to 5 books at a time for a 14-day duration.
Question: How many books can undergrads borrow?
Answer: According to [Source 1, Page 4], undergraduate students are allowed to borrow up to 5 books at a time for a duration of 14 days.

### EXAMPLE 2 (Information absent from context):
Context:
[Source 1 | File: Policy.pdf | Page: 2]
The cafeteria operates from 8:00 AM to 8:00 PM on weekdays.
Question: Does the university provide free airport pickup?
Answer: I could not find this information in the uploaded documents.

### EXAMPLE 3 (Partial information present):
Context:
[Source 1 | File: Syllabus.pdf | Page: 3]
The final exam will cover Modules 1 through 4 and is worth 30% of the final grade.
Question: What is the date and time of the final exam?
Answer: The provided document mentions that the final exam covers Modules 1 through 4 and carries a 30% grade weight [Source 1, Page 3], but the specific date and time of the exam are not mentioned in the uploaded documents.

---
NOW ANSWER THE FOLLOWING:

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""

# 4. Executive Briefing & Structured Matrix Prompting
EXECUTIVE_RAG_PROMPT = """You are an Executive Intelligence Analyst preparing an audited briefing based on documentation.

INSTRUCTIONS:
1. Synthesize the findings into an Executive Briefing format.
2. Structure your response into:
   - **Executive Summary**: 1-2 sentence core answer.
   - **Key Extracted Facts**: Bulleted list of factual statements with [Source N, Page P] citations.
   - **Evidence Matrix**: A Markdown table summarizing columns | Source | Page | Key Evidence |.
   - **Scope & Missing Information**: Explicit mention of any requested details not found in the context.
3. If no relevant info exists in context, state: "I could not find this information in the uploaded documents."

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION: {question}

EXECUTIVE BRIEFING:"""

# 5. Custom / Dynamic Persona Prompting
CUSTOM_RAG_PROMPT = """{custom_instructions}

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION: {question}

ANSWER:"""

PROMPT_STRATEGIES = {
    "Strict Grounding (Anti-Hallucination)": STRICT_RAG_PROMPT,
    "Evidence-First Answer (Cited Facts)": EVIDENCE_FIRST_RAG_PROMPT,
    "Few-Shot Exemplar (In-Context Demonstrations)": FEW_SHOT_RAG_PROMPT,
    "Executive Briefing & Matrix": EXECUTIVE_RAG_PROMPT,
    "Custom Persona / Prompt": CUSTOM_RAG_PROMPT,
}


def _is_valid_key(key: Optional[str]) -> bool:
    """Check if an API key is present and not a dummy placeholder."""
    if not key:
        return False
    key = key.strip()
    placeholder_tokens = ["your_", "api_key_here", "placeholder", "<key>"]
    if any(token in key.lower() for token in placeholder_tokens):
        return False
    return len(key) > 8


def _get_embedding_model(api_key: Optional[str] = None):
    """
    Return the embedding model.
    Priority:
      1. Google Generative AI Embeddings (if valid GOOGLE_API_KEY available)
      2. HuggingFace sentence-transformers (offline, free, no API key needed)
    """
    google_key = api_key if _is_valid_key(api_key) else os.getenv("GOOGLE_API_KEY")
    if _is_valid_key(google_key):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_key,
            )
        except Exception:
            pass

    # Local development fallback: available only when sentence-transformers is installed.
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as exc:
        raise RuntimeError(
            "No embedding model is available. Add GOOGLE_API_KEY in Hugging Face "
            "Space secrets, or install sentence-transformers for local offline use."
        ) from exc


def _get_llm(api_key: Optional[str] = None, provider_override: Optional[str] = None):
    """
    Return the LLM (Language Model) used to generate answers.

    Priority order:
      1. NVIDIA API Catalog model (NVIDIA_API_KEY – OpenAI-compatible endpoint)
      2. Google Gemini         (GOOGLE_API_KEY)
      3. OpenAI GPT            (OPENAI_API_KEY)
      4. Groq                  (GROQ_API_KEY)
      5. Ollama                (local, if selected or running)
    """
    provider_override = None if provider_override in (None, "Auto-detect") else provider_override

    nvidia_key = api_key if (provider_override == "NVIDIA Nemotron" and _is_valid_key(api_key)) else os.getenv("NVIDIA_API_KEY")
    google_key = api_key if (provider_override == "Google Gemini" and _is_valid_key(api_key)) else os.getenv("GOOGLE_API_KEY")
    openai_key = api_key if (provider_override == "OpenAI" and _is_valid_key(api_key)) else os.getenv("OPENAI_API_KEY")
    groq_key   = api_key if (provider_override == "Groq" and _is_valid_key(api_key)) else os.getenv("GROQ_API_KEY")

    # Auto-detect key type from prefix when no provider override given
    if _is_valid_key(api_key) and not provider_override:
        if api_key.startswith("nvapi-"):
            nvidia_key = api_key
        elif api_key.startswith("AIza"):
            google_key = api_key
        elif api_key.startswith("sk-"):
            openai_key = api_key
        elif api_key.startswith("gsk_"):
            groq_key = api_key
        else:
            google_key = api_key

    # 1. NVIDIA API (uses OpenAI-compatible endpoint)
    if _is_valid_key(nvidia_key):
        from langchain_openai import ChatOpenAI
        # meta/llama-3.2-11b-vision-instruct is confirmed working on free-tier NVIDIA API keys.
        # nemotron-ultra-253b requires an additional subscription unlock on the NVIDIA account.
        return ChatOpenAI(
            model="meta/llama-3.2-11b-vision-instruct",
            openai_api_key=nvidia_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            max_tokens=2048,
        )

    # 2. Google Gemini
    if _is_valid_key(google_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_key,
            temperature=0.1,
            convert_system_message_to_human=True,
        )

    # 3. OpenAI
    if _is_valid_key(openai_key):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=openai_key,
            temperature=0.1,
        )

    # 4. Groq
    if _is_valid_key(groq_key):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_key,
            temperature=0.1,
        )

    # 5. Ollama local LLM fallback
    if provider_override == "Ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="llama3", temperature=0.1)
        except Exception as e:
            raise RuntimeError(f"Could not connect to Ollama: {e}")

    raise RuntimeError(
        "No LLM API key detected.\n\n"
        "To enable answer generation:\n"
        "1. Open the 'Model, API & Chunking Configuration' panel and paste your NVIDIA / Google / OpenAI / Groq key,\n"
        "   OR\n"
        "2. Add one supported key to the `.env` file, for example GOOGLE_API_KEY=your_key."
    )


# Module-level vector store instance
_vector_store: Optional[FAISS] = None


def build_vector_store(chunks: List[Document], api_key: Optional[str] = None) -> None:
    """
    Create a FAISS vector store from document chunks.
    """
    global _vector_store

    if not chunks:
        raise ValueError("No chunks provided. Cannot build vector store.")

    embeddings = _get_embedding_model(api_key=api_key)
    _vector_store = FAISS.from_documents(chunks, embeddings)


def assemble_prompt(
    question: str,
    context: str,
    strategy: str = "Strict Grounding (Anti-Hallucination)",
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Assemble the prompt string for the LLM based on chosen professional prompt strategy.
    Demonstrates Exercise 2: Professional Prompt Design.
    """
    if strategy == "Custom Persona / Prompt":
        instr = (
            custom_instructions.strip()
            if (custom_instructions and custom_instructions.strip())
            else "You are an AI assistant. Answer the question using the context below."
        )
        return CUSTOM_RAG_PROMPT.format(custom_instructions=instr, context=context, question=question)

    template = PROMPT_STRATEGIES.get(strategy, STRICT_RAG_PROMPT)
    return template.format(context=context, question=question)


def answer_question(
    question: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    strategy: str = "Strict Grounding (Anti-Hallucination)",
    custom_instructions: Optional[str] = None,
    top_k: int = TOP_K,
) -> Tuple[str, List[Document], str]:
    """
    Run the full RAG pipeline for a user question:
      question
        -> FAISS similarity search with score (returns top_k chunks)
        -> format context from chunks with metadata & score
        -> inject into selected prompt template (Exercise 2)
        -> LLM generates grounded answer
      returns (answer_text, source_documents, assembled_prompt)
    """
    global _vector_store

    if _vector_store is None:
        raise RuntimeError(
            "No documents have been processed yet.\n"
            "Please upload documents and click 'Process Documents' first."
        )

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty. Please type a question.")

    # 1. Similarity search with score (L2 distance or cosine metric)
    results = _vector_store.similarity_search_with_score(question, k=int(top_k))

    if not results:
        return "I could not find this information in the uploaded documents.", [], ""

    sources: List[Document] = []
    context_chunks: List[str] = []

    for i, (doc, score) in enumerate(results, 1):
        doc.metadata["similarity_score"] = float(score)
        sources.append(doc)

        src = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", 0) + 1
        context_chunks.append(
            f"[Source {i} | File: {src} | Page: {page} | Distance: {score:.4f}]\n"
            f"{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_chunks)

    # 2. Assemble prompt using selected strategy (Exercise 2)
    prompt_text = assemble_prompt(
        question=question,
        context=context,
        strategy=strategy,
        custom_instructions=custom_instructions,
    )

    # 3. Query LLM
    llm = _get_llm(api_key=api_key, provider_override=provider)
    response = llm.invoke(prompt_text)

    # 4. Extract answer text
    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    return answer.strip(), sources, prompt_text


def format_sources(sources: List[Document]) -> str:
    """Format retrieved source documents into clean markdown with similarity scores."""
    if not sources:
        return "_No source documents were retrieved._"

    lines = []
    for i, doc in enumerate(sources, start=1):
        meta      = doc.metadata
        file_name = meta.get("source", "Unknown file")
        page_num  = meta.get("page", 0) + 1
        score     = meta.get("similarity_score")
        snippet   = doc.page_content.strip()

        score_info = f" • 🎯 **FAISS Distance:** `{score:.4f}`" if score is not None else ""

        if len(snippet) > 400:
            snippet = snippet[:400] + "..."

        lines.append(
            f"**Source {i}**\n"
            f"- 📁 **File:** `{file_name}` | 📄 **Page:** {page_num}{score_info}\n\n"
            f"> {snippet}\n"
        )

    return "\n---\n".join(lines)


def is_ready() -> bool:
    """Return True if the vector store has been built and is ready for queries."""
    return _vector_store is not None


def reset() -> None:
    """Clear the in-memory vector store."""
    global _vector_store
    _vector_store = None
