"""Single-pass RAG (SQL retrieval, then answer generation) and the agent runner used by run.py."""

from generator.answer import extractive_answer, generate_answer
from generator.client import llm_is_unavailable
from retriever.agent import run_agent
from retriever.school_aliases import detect_school_ids
from retriever.search.fulltext_search import fulltext_search
from retriever.search.sql_search import sql_search


def run_rag_pipeline(query: str) -> bool:
    """Run one text-to-SQL retrieval + answer generation (without the LangGraph agent)."""
    print("\nStarting the RAG flow")
    print(f"   Question: '{query}'")

    print("  [RAG] Querying the database...")
    results, sql = sql_search(query)

    retrieval_method = "SQL"
    if not results:
        print("  [RAG] SQL search unavailable or empty; trying full-text search...")
        school_ids = detect_school_ids(query)
        results = fulltext_search(
            query,
            school_id=school_ids[0] if school_ids else None,
            limit=5,
        )
        retrieval_method = "full-text"
        if not results:
            print("  [RAG] No matching database content was found.")
            return False

    print(f"  [RAG] retrieval: {retrieval_method}" + (f"; SQL: {sql}" if sql else ""))
    print(f"  [RAG] Got {len(results)} items")

    print("  [RAG] Generating the answer with OpenAI...")
    answer = None if llm_is_unavailable() else generate_answer(query, results)
    if not answer:
        print("  [RAG] LLM unavailable; returning grounded database excerpts.")
        answer = extractive_answer(results)

    print("\n" + "=" * 30 + " OpenAI answer " + "=" * 30)
    print(answer)
    print("=" * 73 + "\n")
    return True


def run_agent_pipeline(query: str, max_steps: int = 5, verbose: bool = True) -> bool:
    """Run the Agentic RAG flow (LangGraph)."""
    answer = run_agent(query, max_steps=max_steps, verbose=verbose)
    if answer:
        print("\n" + "=" * 30 + " LangGraph Agent answer " + "=" * 30)
        print(answer)
        print("=" * 73 + "\n")
        return True
    print("Failed to generate an answer.")
    return False
