"""
test_prompt_and_rag.py
-----------------------
Verifies:
  1. PDF & Text document ingestion with metadata.
  2. All 5 professional prompt strategies.
  3. FAISS similarity search with score.
  4. Prompt assembly and context injection.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import document_processor as dp
import rag_engine as rag


def test_suite():
    print("=" * 65)
    print("RUNNING EXERCISE 1 & EXERCISE 2 AUTOMATED TEST SUITE")
    print("=" * 65)

    # Step 1: Ingestion Test
    pdf_path = "data/sample_documents/AI_Course_Syllabus.pdf"
    assert os.path.exists(pdf_path), f"Missing {pdf_path}"

    chunks, stats = dp.load_and_split_pdfs([pdf_path], chunk_size=500, chunk_overlap=100)
    print(f"[PASS] Exercise 1 Ingestion: {len(chunks)} chunks created from '{pdf_path}'")
    assert len(chunks) > 0

    # Step 2: FAISS Indexing Test
    rag.build_vector_store(chunks)
    assert rag.is_ready()
    print("[PASS] Exercise 1 Retrieval: FAISS vector database ready")

    # Step 3: Similarity Search with Score
    query = "What are the prerequisites for this course?"
    results = rag._vector_store.similarity_search_with_score(query, k=3)
    assert len(results) > 0
    top_doc, top_score = results[0]
    print(f"[PASS] Similarity Search with Score: Top L2 Distance = {top_score:.4f}")
    print(f"       Top Match Excerpt: {top_doc.page_content[:80]}...")

    # Step 4: Exercise 2 Prompt Design Strategies Test
    context_mock = "[Source 1 | File: AI_Course_Syllabus.pdf | Page: 1]\nPrerequisites: CS201 and Linear Algebra."
    strategies = list(rag.PROMPT_STRATEGIES.keys())

    for strat in strategies:
        assembled = rag.assemble_prompt(
            question="What are the prerequisites?",
            context=context_mock,
            strategy=strat,
            custom_instructions="You are an Academic Director. Be concise.",
        )
        assert "{context}" not in assembled, f"Failed formatting in {strat}"
        assert "{question}" not in assembled, f"Failed formatting in {strat}"
        assert "CS201" in assembled, f"Context missing in {strat}"
        print(f"[PASS] Exercise 2 Prompt Design: Strategy '{strat}' successfully assembled.")

    # Step 5: Format Sources Test
    top_doc.metadata["similarity_score"] = top_score
    formatted = rag.format_sources([top_doc])
    assert "FAISS Distance" in formatted
    print("[PASS] Source Formatting with Distance Score verified.")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! (EXERCISE 1 & 2 VERIFIED)")
    print("=" * 65)


if __name__ == "__main__":
    test_suite()
