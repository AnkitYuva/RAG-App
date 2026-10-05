---
title: Sem1 RAG Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
---

# 📚 Ask My Documents – RAG Assistant & Prompt Engineering System

> **Agentic AI – Internal Assessment (30 Marks Practical Assessment)**
> - **Exercise 1 – RAG Application with UI** (10 Marks)
> - **Exercise 2 – Professional Prompt Design** (10 Marks)
> - **Exercise 3 – Cloud Deployment from GitHub** (10 Marks)

---

## 📖 Key Documentation Quick-Links
- 🚀 **[Deployment Guide (Exercise 3)](DEPLOYMENT_GUIDE.md)**: Push to GitHub and deploy to Hugging Face Spaces / Render in under 3 minutes.
- 🎓 **[Viva & Technical Explanation Guide](VIVA_EXPLANATION_GUIDE.md)**: Architecture, workflow, tools/libraries reasoning, and 15 expected viva questions with answers.
- 🧪 **[Automated Test Verification](test_prompt_and_rag.py)**: End-to-end verification script for ingestion, FAISS indexing, similarity retrieval, and prompt strategies.

---

## 📌 Problem Statement

Large Language Models (LLMs) like GPT and Gemini are trained on general internet
data. They have two critical limitations when used in specialized domains:

1. **Knowledge cutoff** – they don't know about documents published after their
   training date.
2. **Hallucination** – when they don't know an answer, they sometimes invent one.

**RAG solves both problems.** Instead of relying on the LLM's memorized knowledge,
RAG retrieves the actual relevant text from *your* documents and shows it to the
LLM. The LLM then synthesizes an answer *grounded* in your documents.

This application demonstrates a complete, working RAG pipeline with:
- **Exercise 1:** Multi-format document ingestion (PDF, TXT, MD), FAISS vector similarity search, and an interactive Gradio UI.
- **Exercise 2:** 5 professional prompt engineering strategies (Strict Anti-Hallucination, Evidence-First, Few-Shot, Executive Matrix, and Custom Persona) + Live Prompt Inspector.
- **Exercise 3:** Complete deployment readiness for Hugging Face Spaces, Render, and Docker.

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────┐
                         │         USER INTERFACE           │
                         │          (Gradio UI)             │
                         └──────────┬──────────────┬────────┘
                                    │              │
                              Upload PDF        Ask Question
                                    │              │
                    ┌───────────────▼──────┐       │
                    │   PyPDFLoader        │       │
                    │   (Load Pages)       │       │
                    └───────────────┬──────┘       │
                                    │              │
                    ┌───────────────▼──────┐       │
                    │ RecursiveCharacter   │       │
                    │ TextSplitter         │       │
                    │ (Create Chunks)      │       │
                    └───────────────┬──────┘       │
                                    │              │
                    ┌───────────────▼──────┐       │
                    │  Embedding Model     │◄──────┤
                    │  (Text → Vectors)    │       │
                    └───────────────┬──────┘       │
                                    │              │
                    ┌───────────────▼──────┐       │
                    │   FAISS Vector DB    │       │
                    │   (Store Vectors)    │◄──────┘
                    └───────────────┬──────┘
                                    │ Top-K Similar Chunks
                    ┌───────────────▼──────┐
                    │   LLM (Gemini/GPT)   │
                    │   (Generate Answer)  │
                    └───────────────┬──────┘
                                    │
                    ┌───────────────▼──────┐
                    │  Answer + Sources    │
                    │  (Displayed in UI)   │
                    └──────────────────────┘
```

### Component Breakdown

| Component | File | Purpose |
|---|---|---|
| **Document Processor** | `document_processor.py` | PDF loading + text chunking |
| **RAG Engine** | `rag_engine.py` | Embeddings + FAISS + LLM chain |
| **Web Interface** | `app.py` | Gradio UI + event handling |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Gradio** | Web UI framework |
| **LangChain** | RAG pipeline orchestration |
| **PyPDF** | PDF text extraction |
| **FAISS** | Vector similarity search database |
| **Google Gemini** | LLM + Embeddings (default) |
| **Sentence Transformers** | Offline embedding fallback |
| **python-dotenv** | API key management |

---

## 🚀 Installation

### 1. Prerequisites

- Python 3.10 or newer
- pip

### 2. Clone / Download the project

```bash
# If using Git:
git clone <your-repo-url>
cd rag-gradio-app

# Or simply navigate to the project folder:
cd path/to/rag-gradio-app
```

### 3. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows users:** If `faiss-cpu` installation fails, try:
> ```bash
> pip install faiss-cpu --no-cache-dir
> ```

---

## ⚙️ Configuration

### Step 1: Copy the environment template

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### Step 2: Add your API key

Open the `.env` file and fill in **one** of the following:

```env
# Option A – Google Gemini (FREE tier available at https://aistudio.google.com)
GOOGLE_API_KEY=your_google_api_key_here

# Option B – OpenAI GPT
OPENAI_API_KEY=your_openai_api_key_here

# Option C – Groq (free, fast LLaMA models at https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Security:** Never commit your `.env` file to Git.  
> The `.gitignore` already excludes it.

### Which LLM to use?

| Provider | Model Used | Cost | Speed |
|---|---|---|---|
| **Google Gemini** | `gemini-1.5-flash` | Free tier | Fast |
| **NVIDIA API Catalog** | `meta/llama-3.2-11b-vision-instruct` | Free/paid depending on account | Fast |
| **OpenAI** | `gpt-3.5-turbo` | Paid | Fast |
| **Groq** | `llama-3.1-8b-instant` | Free tier | Very fast |

**Recommendation for students:** Use **Google Gemini** – it has a generous free
tier and requires no payment information.

---

## ▶️ Running the Application

```bash
python app.py
```

The application will open automatically in your browser at:  
**http://127.0.0.1:7860**

---

## 📖 How to Use

1. **Upload documents** – Click the upload area and select one or more PDF, TXT, or Markdown files.
2. **Process Documents** – Click "Process Documents". Wait for the status to show
   the number of chunks created.
3. **Ask a Question** – Type your question and click "ASK" (or press Enter).
4. **Review the Answer** – The answer and the source documents used are displayed.

---

## 📁 Project Structure

```
rag-gradio-app/
│
├── app.py                   # Main entry point – Gradio UI
├── document_processor.py    # PDF loading + text chunking
├── rag_engine.py            # Embeddings + FAISS + LLM chain
│
├── requirements.txt         # Python dependencies
├── .env.example             # API key template (copy → .env)
├── .env                     # Your actual API key (DO NOT COMMIT)
├── .gitignore               # Excludes sensitive files from Git
├── README.md                # This file
│
├── data/
│   └── sample_documents/    # Place sample PDFs here for testing
│
└── vectorstore/             # (Created at runtime – FAISS index)
```

---

## 🧪 RAG Evaluation: Sample Test Questions

Use these 5 test cases to evaluate your RAG system. Upload a relevant document
(e.g., a university syllabus or research paper) and run each question.

### Test Case 1 – Direct Answer Present
**Question:** *"What are the course objectives?"*  
**Expected:** Clear list of objectives from the syllabus  
**Grounded?** ✅ Yes – answer should cite the objectives section

### Test Case 2 – Multi-Section Answer
**Question:** *"What topics are covered in the first and last units?"*  
**Expected:** Topics from both Unit 1 and the final unit  
**Grounded?** ✅ Yes – requires chunks from multiple pages

### Test Case 3 – Not in Document
**Question:** *"What is the price of the textbook?"*  
**Expected:** "I could not find this information in the uploaded documents."  
**Grounded?** ✅ Yes – LLM should NOT invent a price

### Test Case 4 – Paraphrased Question
**Question:** *"What competencies will students develop?"* (paraphrase of "learning outcomes")  
**Expected:** Same as asking about learning outcomes  
**Grounded?** ✅ Yes – semantic similarity handles paraphrasing

### Test Case 5 – Irrelevant Information in Question
**Question:** *"If a student enjoys cricket, what assignment must they submit?"*  
**Expected:** Either the assignment information OR "not found"  
**Grounded?** ✅ Yes – irrelevant "cricket" context should be ignored

### Evaluation Table

| # | Question | Expected | Retrieved? | Answer Grounded? |
|---|---|---|---|---|
| 1 | Course objectives | Objectives list | ✅ | ✅ |
| 2 | First + last unit topics | Two sections | ✅ | ✅ |
| 3 | Textbook price | "Not found" | ❌ | ✅ |
| 4 | Competencies (paraphrase) | Learning outcomes | ✅ | ✅ |
| 5 | Cricket + assignment | Assignment info | ✅ | ✅ |

---

## 🧠 Key Concepts – Student Explanation Guide

### 1. What is RAG?
**Retrieval-Augmented Generation** is a technique that combines a retrieval
system (search) with a generative LLM. Instead of asking the LLM to answer
from memory, we first *retrieve* relevant passages from a document collection,
then show those passages to the LLM as context.

### 2. Why do we need RAG?
LLMs have two problems: (1) they don't know about private/recent documents,
(2) they sometimes hallucinate. RAG fixes both by grounding answers in real text.

### 3. What is Document Chunking?
Breaking a large document into smaller pieces (e.g., 800 characters) so each
piece can be individually embedded and retrieved. Chunk overlap ensures
boundary information is not lost.

### 4. What are Embeddings?
Dense numerical vectors (lists of numbers) that encode the *meaning* of text.
The embedding model converts text into a vector so that similar meanings
produce similar vectors in high-dimensional space.

### 5. What is a Vector Database?
A database optimized for storing and searching vectors. Instead of exact
keyword matching, it finds vectors that are *closest* to a query vector.

### 6. What is FAISS?
**Facebook AI Similarity Search** – a library that stores vectors in an index
and supports extremely fast nearest-neighbour search, even for millions of vectors.

### 7. What is Similarity Search?
Given a query vector (embedded question), find the k vectors in the database
that are most similar (smallest cosine or Euclidean distance).

### 8. What is Top-K Retrieval?
Returning the K most similar chunks (default K=4). More K = more context
for the LLM, but longer prompts and higher cost.

### 9. What is the role of the LLM?
The LLM reads the retrieved chunks and synthesizes a coherent answer. It acts
as the "reader and writer" – it does NOT search the database.

### 10. Why display sources?
Source display proves that the answer is grounded in real document text, not
invented. This is called *explainability* – users can verify the answer.

---

## 🔮 Future Enhancements

| Enhancement | Description |
|---|---|
| **Chat History** | Maintain conversation memory across multiple questions |
| **Multiple Collections** | Separate vector stores per document set |
| **Reranking** | Use a cross-encoder to re-rank retrieved chunks |
| **Hybrid Search** | Combine keyword (BM25) + semantic search |
| **Source Highlighting** | Highlight exact sentences in the PDF viewer |
| **Streaming Responses** | Stream the LLM answer token-by-token |
| **Authentication** | Add user login for private deployments |
| **Persistent Store** | Save FAISS index to disk for reuse across sessions |
| **Different Embeddings** | Experiment with OpenAI/Cohere/local embeddings |
| **Metadata Filtering** | Filter chunks by document name, date, etc. |

---

## ✅ Internal Assessment Mapping

| Assessment Component | Where It Is Implemented |
|---|---|
| **Exercise 1 – RAG Application with UI** | `app.py`, `document_processor.py`, and `rag_engine.py` implement upload, processing, FAISS retrieval, answer generation, and source display. |
| **Exercise 2 – Professional Prompt Design** | The prompt strategy dropdown and live prompt inspector demonstrate strict grounding, evidence-first answers, few-shot examples, executive formatting, and custom persona prompting. |
| **Exercise 3 – Deployment** | `DEPLOYMENT_GUIDE.md`, `Dockerfile`, `Procfile`, and this README explain GitHub deployment to Hugging Face Spaces or Render. |

---

## 📜 License

This project is created for educational purposes.  
Feel free to use, modify, and distribute it for learning.
