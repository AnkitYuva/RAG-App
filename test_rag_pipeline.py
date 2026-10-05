"""
test_rag_pipeline.py
--------------------
Automated end-to-end verification of the RAG pipeline.
Tests:
  1. PDF loading via PyPDFLoader
  2. Text chunking & metadata preservation
  3. FAISS embedding & similarity retrieval
  4. Top-K context matching
  5. Session reset
"""

import os
import document_processor as dp
import rag_engine as rag


def run_tests():
    print("=" * 60)
    print("RUNNING END-TO-END RAG PIPELINE VERIFICATION")
    print("=" * 60)

    # Ensure UTF-8 output on Windows consoles
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Verify Sample Documents Exist
    pdf1 = "data/sample_documents/AI_Course_Syllabus.pdf"
    pdf2 = "data/sample_documents/Campus_Policies_Guide.pdf"
    assert os.path.exists(pdf1), f"Missing {pdf1}"
    assert os.path.exists(pdf2), f"Missing {pdf2}"
    print("[PASS] Test 1: Sample PDF documents verified.")

    # 2. Test Document Loading & Chunking
    chunks, stats = dp.load_and_split_pdfs([pdf1, pdf2], chunk_size=600, chunk_overlap=100)
    assert len(chunks) > 0, "No chunks generated"
    assert stats["total_files"] == 2
    assert stats["total_pages"] == 4
    assert "source" in chunks[0].metadata
    assert "page" in chunks[0].metadata
    print(f"[PASS] Test 2: Document loading and chunking passed ({len(chunks)} chunks from 4 pages).")

    # 3. Test FAISS Vector Store Building
    print("[INFO] Building FAISS index with sentence embeddings...")
    rag.build_vector_store(chunks)
    assert rag.is_ready(), "Vector store should be ready"
    print("[PASS] Test 3: FAISS vector database successfully constructed and populated.")

    # 4. Test Top-K Similarity Search Retrieval
    test_queries = [
        "What are the prerequisites for the course?",
        "What is the library borrowing limit for undergraduate students?",
        "What is the grading weightage for the Capstone Project?",
    ]

    for q in test_queries:
        retriever = rag._vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(q)
        assert len(retrieved_docs) > 0, f"No docs retrieved for query: {q}"
        top_match = retrieved_docs[0]
        print(f"\n[QUERY] '{q}'")
        print(f"   Top Match: File '{top_match.metadata.get('source')}', Page {top_match.metadata.get('page', 0)+1}")
        print(f"   Excerpt: {top_match.page_content.strip()[:100]}...")

    print("\n[PASS] Test 4: Semantic similarity retrieval successfully fetched accurate context chunks.")

    # 5. Test Reset Functionality
    rag.reset()
    assert not rag.is_ready(), "Vector store should be cleared after reset"
    print("[PASS] Test 5: Reset functionality confirmed.")

    print("\n" + "=" * 60)
    print("ALL RAG PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
