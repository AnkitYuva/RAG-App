# 🎓 Viva & Oral Examination Guide
## Agentic AI Internal Assessment: RAG Architecture & Implementation

> **Objective:** This guide provides complete conceptual clarity, technical reasoning, and sample viva Q&A to help you confidently explain **how and why** your solution works to the faculty.

---

## 1. ⚡ 30-Second Elevator Pitch (Start with this)

> *"Sir/Ma'am, our project is an enterprise-grade Retrieval-Augmented Generation (RAG) assistant. It bridges the gap between static LLMs and private documents. Standard LLMs suffer from knowledge cutoffs and hallucinations. Our application ingests proprietary PDF, TXT, and Markdown files, decomposes them into overlapping semantic chunks, indexes them into a high-dimensional FAISS vector database, retrieves the most relevant chunks via semantic similarity scoring, injects them into professionally engineered prompt templates with anti-hallucination guardrails, and produces grounded, verifiable answers with exact source and page citations. The system is containerized and deployed from GitHub to the cloud."*

---

## 2. 🏗️ Architecture & End-to-End Workflow

```
[User Document: PDF / TXT / MD]
               │
               ▼
   [1. Ingestion: PyPDFLoader]
   • Parses raw binary data into text pages
   • Extracts metadata: filename, page number
               │
               ▼
   [2. Chunking: RecursiveCharacterTextSplitter]
   • Chunk size: 800 chars | Overlap: 150 chars
   • Hierarchical splits: ["\n\n", "\n", ". ", " ", ""]
               │
               ▼
   [3. Embeddings: Dense Vector Transformation]
   • Google text-embedding-004 or HuggingFace all-MiniLM-L6-v2 (384-dim)
   • Converts natural language chunks into mathematical vectors
               │
               ▼
   [4. Indexing: FAISS Vector Database]
   • Indexes high-dimensional embeddings for sub-millisecond nearest neighbor search
               │
               ├─────────────────────────────────────────────┐
               ▼                                             │
      [User Submits Query]                                   │
               │                                             │
               ▼                                             │
   [5. Query Embedding & Similarity Search]                  │
   • Calculates L2 distance / Cosine similarity              │
   • Fetches Top-K most relevant chunks (e.g. k=4)           │
               │                                             │
               ▼                                             │
   [6. Exercise 2: Professional Prompt Assembly]             │
   • Injects retrieved chunks into selected strategy         │
   • Applies anti-hallucination guardrails & citation rules  │
               │                                             │
               ▼                                             │
   [7. Grounded Generation: LLM]                             │
   • Gemini 1.5 Flash / GPT-3.5 / LLaMA 3.1                  │
   • Synthesizes verified answer                             │
               │                                             │
               ▼                                             │
   [8. User Interface Display (Gradio)] ◄────────────────────┘
   • Displays Answer + Citations (File, Page, FAISS Score)
   • Live Assembled Prompt Inspector for live verification
```

---

## 3. 🛠️ Tools & Libraries Used (The "Why")

| Library / Tool | Role in Project | Why We Chose It (Engineering Rationale) |
|---|---|---|
| **Python 3.10+** | Core runtime | Industry standard for AI/ML and LangChain ecosystems. |
| **LangChain Core & Community** | Orchestration framework | Standardizes document loaders, text splitters, vector stores, and prompt templates into composable pipelines. |
| **PyPDF** | PDF parsing | Fast, lightweight extraction of text and per-page metadata without requiring heavyweight OCR engines. |
| **FAISS (CPU)** | Vector database | Developed by Facebook AI Research; optimized in C++ for ultra-fast nearest-neighbor similarity search. Runs locally with zero cloud database overhead. |
| **Sentence-Transformers (`all-MiniLM-L6-v2`)** | Embedding model | 384-dimensional dense vectors. Runs locally on CPU for offline fallback with high semantic fidelity. |
| **Google Gemini (`text-embedding-004` & `gemini-1.5-flash`)** | Cloud Embeddings & LLM | State-of-the-art token efficiency, low latency, and a free tier accessible for educational demos. |
| **Gradio** | Web UI framework | Rapid, responsive, interactive UI designed specifically for machine learning apps with state management and event-driven architecture. |
| **Docker** | Containerization | Packages OS dependencies, Python packages, and application code into a single immutable image for universal deployment. |

---

## 4. 🧠 Deep Dive into Exercise 2: Professional Prompt Design

A key evaluation criterion is **"clarity, appropriate context handling, and the ability to generate accurate and relevant responses."**

We implemented **5 distinct prompt engineering strategies** that you can toggle in real-time in the UI:

### Strategy 1: Strict Grounding (Anti-Hallucination & Defensiveness)
- **Role:** Eliminates hallucination by enforcing a closed-world assumption.
- **Key Techniques Used:**
  1. *Negative Constraint:* "Rely ONLY on the explicitly stated facts in the DOCUMENT CONTEXT. Do NOT assume, extrapolate, or bring in outside knowledge."
  2. *Deterministic Refusal Trigger:* "If the answer cannot be found in the provided context, respond with: 'I could not find this information in the uploaded documents.'"
  3. *Citation Formatting Rule:* Requires citations in `[Source N, Page P]` brackets.

### Strategy 2: Evidence-First Answer (Cited Facts)
- **Role:** Makes the model show the retrieved facts it used before giving the final answer.
- **Key Techniques Used:**
  - Extracts only facts that appear in the retrieved document chunks.
  - Checks whether those facts are sufficient to answer the question.
  - Generates two explicit sections: `**Evidence Used:**` and `**Final Answer:**`.

### Strategy 3: Few-Shot In-Context Exemplar Prompting
- **Role:** Demonstrates desired input-output behaviors using concrete examples.
- **Key Techniques Used:**
  - Example 1: Direct factual match.
  - Example 2: Complete absence of facts (demonstrating graceful refusal).
  - Example 3: Partial information (answering what is known and identifying what is missing).

### Strategy 4: Executive Briefing & Evidence Matrix
- **Role:** Generates an enterprise-ready summary with tabular evidence.
- **Key Techniques Used:**
  - Outputs an Executive Summary, Key Extracted Facts, and a Markdown table (`| Source | Page | Key Evidence |`).

### Strategy 5: Custom Persona & Live Prompt Inspector
- **Role:** Demonstrates dynamic persona injection (e.g. Academic Auditor, Legal Reviewer) and provides an interactive accordion showing the **full assembled prompt** sent to the model.

---

## 5. ❓ Top 15 Viva Questions & Model Answers

### Q1: What is RAG, and why is it needed when we already have powerful LLMs?
**Answer:** RAG stands for Retrieval-Augmented Generation. LLMs have two fundamental weaknesses: **knowledge cutoff** (they cannot know private or recent documents) and **hallucination** (they invent plausible-sounding but false answers). RAG solves this by retrieving relevant text chunks from our private documents and injecting them as explicit context into the prompt, forcing the LLM to answer based on verified facts rather than memorized weights.

### Q2: Why do we split documents into chunks instead of passing the entire document?
**Answer:**
1. **Context Window Limits:** Passing entire multi-page documents exceeds model context limits or incurs heavy token costs.
2. **Signal-to-Noise Ratio ("Lost in the Middle"):** LLMs attend best to compact, focused context. Large documents dilute relevance and increase hallucination risk.
3. **Retrieval Efficiency:** Vector databases search small semantic paragraphs much more accurately than 50-page files.

### Q3: What is Chunk Overlap and why is it important?
**Answer:** Chunk overlap (we used 150 characters) ensures that text across chunk boundaries is not severed. If a crucial sentence or definition begins at the end of Chunk A and finishes in Chunk B, overlap guarantees that at least one chunk captures the complete semantic thought.

### Q4: How does `RecursiveCharacterTextSplitter` work?
**Answer:** It recursively splits text using a priority list of delimiters: double newlines (`\n\n` for paragraphs), single newlines (`\n` for lines), spaces (` ` for words), and finally characters. This ensures text is broken down at the most natural semantic boundaries possible without arbitrarily cutting words.

### Q5: What is a Vector Embedding?
**Answer:** An embedding is a dense, high-dimensional vector representation of text (e.g., 384 dimensions in MiniLM, or 768/1536 in modern foundation models). Words and sentences with similar semantic meanings are placed close to each other in vector space, allowing mathematical calculation of meaning.

### Q6: What is FAISS and how does it retrieve documents?
**Answer:** FAISS (Facebook AI Similarity Search) is an open-source C++ library optimized for indexing and searching vector embeddings. When a user asks a question, we embed the query into a vector and FAISS computes the distance (e.g., L2 Euclidean distance or Cosine similarity) against all indexed document vectors, returning the top-k nearest neighbors.

### Q7: What does the FAISS distance score shown in your UI mean?
**Answer:** It represents the geometric distance in vector space between the user's query embedding and the chunk's embedding. In L2 distance, a **lower number indicates closer proximity and higher semantic relevance**.

### Q8: What is the purpose of Top-K?
**Answer:** Top-K determines how many relevant chunks are fetched from the vector store to assemble the prompt context. If K is too small (e.g., K=1), we might miss complementary context. If K is too large (e.g., K=15), we flood the prompt with marginally relevant text. We make K configurable (default K=4) to balance completeness and precision.

### Q9: How does your application prevent hallucinations?
**Answer:**
1. **Strict System Prompt Guardrails:** We explicitly instruct the LLM to rely only on the context and forbid outside knowledge.
2. **Defensive Fallback Instruction:** We specify a mandatory fallback sentence: *"I could not find this information in the uploaded documents."*
3. **Mandatory Citations:** The model must reference `[Source N, Page P]`. If it cannot cite a source, it cannot claim the fact.
4. **Low Temperature:** We set the model temperature to `0.1` for deterministic, fact-focused responses.

### Q10: How did you implement professional prompt design?
**Answer:** We implemented multiple prompt templates and allow the user to choose them from the UI. The most important one is strict grounding: it tells the LLM to answer only from retrieved context, cite sources, and use a fixed fallback sentence when the answer is absent. The evidence-first strategy asks the model to list cited facts before the final answer, which makes the answer easier to audit during the viva.

### Q11: How do you handle cases where the document does NOT contain the answer?
**Answer:** In our test cases, we test questions like *"What is the price of the textbooks?"* against the course syllabus. Because price is unmentioned, the model triggers the anti-hallucination fallback instead of making up a dollar figure.

### Q12: Why did you use Gradio instead of standard HTML/Flask?
**Answer:** Gradio is built specifically for machine learning and AI workflows. It provides built-in state management (`gr.State`), native file upload handling, reactive UI components, streaming support, and 1-click cloud deployment on Hugging Face Spaces.

### Q13: How is your application deployed?
**Answer:** We prepared three deployment pathways:
1. **Local:** Running via Python virtual environment on `127.0.0.1:7860`.
2. **Hugging Face Spaces:** Directly linked to our GitHub repository with native Gradio hosting and HTTPS.
3. **Docker:** A containerized `Dockerfile` based on `python:3.10-slim` ready for cloud hosting platforms like Render or AWS.

### Q14: How are API keys secured in your application?
**Answer:** API keys are never hardcoded. They are loaded via `python-dotenv` from a `.env` file that is strictly ignored by `.gitignore`. In deployed cloud environments, keys are passed via secure environment variables/secrets.

### Q15: What would you improve in this RAG pipeline for a production enterprise?
**Answer:**
1. **Hybrid Search:** Combine dense vector search with BM25 sparse keyword search (Reciprocal Rank Fusion).
2. **Cross-Encoder Re-Ranking:** Re-rank top-20 retrieved candidates using a cross-encoder model before feeding the top-4 to the LLM.
3. **Contextual Compression & Chunk Routing:** Filter out irrelevant sentences inside chunks before prompt injection.
4. **Persistent Vector Database:** Transition from in-memory FAISS to a distributed vector database like ChromaDB, Milvus, or Qdrant.
