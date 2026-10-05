"""
create_sample_docs.py
---------------------
Generates realistic sample PDF documents for testing the RAG Assistant.
Creates:
  1. data/sample_documents/AI_Course_Syllabus.pdf
  2. data/sample_documents/Campus_Policies_Guide.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

os.makedirs("data/sample_documents", exist_ok=True)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Title"],
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=12,
)

h1_style = ParagraphStyle(
    "DocH1",
    parent=styles["Heading1"],
    fontSize=14,
    leading=18,
    textColor=colors.HexColor("#0f3460"),
    spaceBefore=14,
    spaceAfter=6,
)

h2_style = ParagraphStyle(
    "DocH2",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#e94560"),
    spaceBefore=10,
    spaceAfter=4,
)

body_style = ParagraphStyle(
    "DocBody",
    parent=styles["BodyText"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#222222"),
    spaceAfter=6,
)

# ─────────────────────────────────────────────────────────────────────────────
# Document 1: AI Course Syllabus
# ─────────────────────────────────────────────────────────────────────────────
doc1_path = "data/sample_documents/AI_Course_Syllabus.pdf"
doc1 = SimpleDocTemplate(doc1_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)

story1 = []
story1.append(Paragraph("Department of Computer Science & Engineering", h2_style))
story1.append(Paragraph("CS601: Applied Artificial Intelligence & RAG Systems", title_style))
story1.append(Paragraph("<b>Semester:</b> Fall 2026 | <b>Instructor:</b> Dr. Rajesh Sharma | <b>Credits:</b> 4.0", body_style))
story1.append(Paragraph("<b>Class Schedule:</b> Mondays and Wednesdays, 10:00 AM – 11:30 AM (Room LH-302)", body_style))
story1.append(Paragraph("<b>Prerequisites:</b> Proficiency in Python programming, linear algebra, and basic probability theory.", body_style))
story1.append(Spacer(1, 10))

story1.append(Paragraph("1. Course Description & Learning Objectives", h1_style))
story1.append(Paragraph(
    "This course provides an end-to-end practical understanding of Modern Artificial Intelligence, "
    "with a core focus on Large Language Models (LLMs), Embeddings, Vector Databases, and Retrieval-Augmented "
    "Generation (RAG). By the end of this course, students will be able to design, implement, and deploy production-ready "
    "RAG systems that eliminate hallucinations and ground LLM responses in proprietary knowledge bases.",
    body_style
))

story1.append(Paragraph("2. Course Modules & Detailed Syllabus", h1_style))
story1.append(Paragraph("<b>Module 1: Foundations of NLP and Tokenization</b>", h2_style))
story1.append(Paragraph(
    "Word embeddings (Word2Vec, GloVe), subword tokenization (BPE, WordPiece), Transformer self-attention mechanism, "
    "and encoder-decoder architectures. Practical exercises include tokenization with Hugging Face.",
    body_style
))

story1.append(Paragraph("<b>Module 2: Large Language Models & Prompt Engineering</b>", h2_style))
story1.append(Paragraph(
    "Autoregressive generation, zero-shot vs few-shot learning, chain-of-thought prompting, temperature and top-p sampling. "
    "Students will work with models like Google Gemini, GPT-4, and LLaMA.",
    body_style
))

story1.append(PageBreak())

story1.append(Paragraph("<b>Module 3: Retrieval-Augmented Generation (RAG) Architecture</b>", h2_style))
story1.append(Paragraph(
    "The fundamental limitations of parametric LLM memory: hallucination, cutoff dates, and domain ignorance. "
    "The RAG paradigm: Document loaders (PyPDFLoader, Unstructured), chunking strategies (recursive character splitting, token-based, semantic chunking), "
    "vector embeddings (Google text-embedding-004, sentence-transformers MiniLM), vector stores (FAISS, Chroma, Pinecone), "
    "and top-k similarity retrieval algorithms (cosine similarity, inner product, L2 distance).",
    body_style
))

story1.append(Paragraph("<b>Module 4: Evaluation, Grounding, and Deployment</b>", h2_style))
story1.append(Paragraph(
    "RAG evaluation metrics: Context Precision, Context Recall, Faithfulness, and Answer Relevance. "
    "Deployment using Gradio, Streamlit, and FastAPI. Edge cases: empty documents, multi-file retrieval, and latency optimization.",
    body_style
))

story1.append(Paragraph("3. Grading Policy & Assessment Scheme", h1_style))
story1.append(Paragraph("The overall course assessment consists of five components:", body_style))

grading_data = [
    ["Assessment Component", "Weightage", "Due Date"],
    ["Lab Assignments (4 in total)", "20%", "Fortnightly"],
    ["Midterm Examination", "25%", "Week 8 (October 24)"],
    ["Capstone RAG Project & Demo", "35%", "Week 14 (December 2)"],
    ["Final Viva Voce", "10%", "Week 15 (December 10)"],
    ["Class Participation & Quizzes", "10%", "Ongoing"],
]
table1 = Table(grading_data, colWidths=[200, 100, 150])
table1.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f3460")),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cccccc")),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
]))
story1.append(table1)

story1.append(Spacer(1, 10))
story1.append(Paragraph("4. Capstone Project Requirements", h1_style))
story1.append(Paragraph(
    "Students must build a complete RAG system with a Gradio web interface named 'Ask My Documents – RAG Assistant'. "
    "The project must demonstrate document loading with PyPDFLoader, chunking with RecursiveCharacterTextSplitter, "
    "embedding generation, FAISS indexing, similarity retrieval, and strict grounded answer generation with source citation. "
    "Late project submissions will incur a 10% penalty per calendar day up to a maximum of 3 days, after which submissions will not be accepted.",
    body_style
))

doc1.build(story1)

# ─────────────────────────────────────────────────────────────────────────────
# Document 2: Campus Policies Guide
# ─────────────────────────────────────────────────────────────────────────────
doc2_path = "data/sample_documents/Campus_Policies_Guide.pdf"
doc2 = SimpleDocTemplate(doc2_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)

story2 = []
story2.append(Paragraph("Apex Institute of Technology", h2_style))
story2.append(Paragraph("Student Campus Policies & Guidelines Handbook (2026)", title_style))
story2.append(Paragraph("<b>Published by:</b> Office of Academic Affairs & Student Welfare", body_style))
story2.append(Paragraph("<b>Applicable to:</b> All Undergraduate, Postgraduate, and Doctoral Students", body_style))
story2.append(Spacer(1, 10))

story2.append(Paragraph("1. Attendance and Leave Policy", h1_style))
story2.append(Paragraph(
    "A minimum attendance of <b>75%</b> is mandatory in each registered course to be eligible to sit for the semester end examinations. "
    "Students with attendance between 65% and 74% may be condoned by the Dean of Academic Affairs only on medical grounds, provided "
    "a valid medical certificate issued by the University Health Centre is submitted within 3 working days of resumption of classes. "
    "Students with attendance below 65% will receive an 'F' grade (Attendance Shortage) and must repeat the course.",
    body_style
))

story2.append(Paragraph("2. Academic Integrity & Plagiarism Regulations", h1_style))
story2.append(Paragraph(
    "The university maintains a zero-tolerance policy towards academic dishonesty, including cheating in examinations, fabrication of data, "
    "and plagiarism. For all assignments and capstone projects, similarity index must not exceed <b>15%</b> (excluding bibliography). "
    "Submitting AI-generated code without explicit disclosure and attribution is considered an academic violation. Penalties for plagiarism "
    "range from zero marks in the assessment for a first offence, to course failure or semester suspension for repeated violations.",
    body_style
))

story2.append(PageBreak())

story2.append(Paragraph("3. Central Library Guidelines & Hours", h1_style))
story2.append(Paragraph(
    "The Central Library is open from <b>8:00 AM to 11:00 PM on weekdays</b>, and from <b>9:00 AM to 6:00 PM on weekends and public holidays</b>. "
    "During end-semester examinations, the reading rooms remain open 24 hours. "
    "Undergraduate students may borrow up to 4 books for a period of 14 days. Postgraduate students may borrow up to 6 books for 28 days. "
    "A late fee of <b>$0.50 (Rs. 10) per day per book</b> is levied on overdue loans.",
    body_style
))

story2.append(Paragraph("4. Computing Laboratories and High-Performance Cluster Access", h1_style))
story2.append(Paragraph(
    "The AI & Deep Learning Lab (Lab 405) is equipped with NVIDIA RTX 4090 GPUs. Access is available Monday through Friday from 9:00 AM to 8:00 PM. "
    "Students requiring GPU cluster time for their capstone projects must submit an allocation request signed by their project guide. "
    "Food and beverages (except sealed water bottles) are strictly prohibited in all computing labs.",
    body_style
))

story2.append(Paragraph("5. Grievance Redressal and Contact Directory", h1_style))
story2.append(Paragraph(
    "Students facing academic or administrative issues may contact the Student Welfare Cell in Administrative Block Room 104, "
    "or email <i>grievance@apex.edu</i>. The Anti-Ragging Helpline operates 24/7 at toll-free number 1800-180-5522.",
    body_style
))

doc2.build(story2)
print("Sample documents generated successfully!")
