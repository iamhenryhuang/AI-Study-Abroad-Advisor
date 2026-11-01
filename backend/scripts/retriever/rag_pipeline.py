# SQL retrieval -> answer generation, or Agentic RAG: the LangGraph flow

import sys
import argparse
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from retriever.sql_search import sql_search
from retriever.fulltext_search import fulltext_search
from retriever.agent import run_agent
from retriever.agent.nodes.finalize import _extractive_fallback_answer
from retriever.agent.state import _detect_school_ids
from generator.answer import generate_answer
from generator.client import llm_is_unavailable


def run_rag_pipeline(query: str) -> bool:
    """Run one text-to-SQL retrieval + answer generation (without the LangGraph agent)."""
    print(f"\nStarting the RAG flow")
    print(f"   Question: '{query}'")

    print(f"  [RAG] Querying the database...")
    results, sql = sql_search(query)

    retrieval_method = "SQL"
    if not results:
        print("  [RAG] SQL search unavailable or empty; trying full-text search...")
        school_ids = _detect_school_ids(query)
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

    print(f"  [RAG] Generating the answer with OpenAI...")
    answer = None if llm_is_unavailable() else generate_answer(query, results)
    if not answer:
        print("  [RAG] LLM unavailable; returning grounded database excerpts.")
        answer = _extractive_fallback_answer(results)

    if answer:
        print("\n" + "=" * 30 + " OpenAI answer " + "=" * 30)
        print(answer)
        print("=" * 73 + "\n")
        return True
    else:
        print("Failed to generate an answer.")
        return False


def run_agent_pipeline(query: str, max_steps: int = 5, verbose: bool = True) -> bool:
    """Run the Agentic RAG flow (LangGraph)."""
    answer = run_agent(query, max_steps=max_steps, verbose=verbose)
    if answer:
        print("\n" + "=" * 30 + " LangGraph Agent answer " + "=" * 30)
        print(answer)
        print("=" * 73 + "\n")
        return True
    else:
        print("Failed to generate an answer.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG flow")
    parser.add_argument("query", nargs="?", help="the user's question")
    parser.add_argument("--agent", action="store_true", help="use Agentic RAG (LangGraph)")
    parser.add_argument("--max-steps", type=int, default=5, help="maximum number of agent iterations")
    args = parser.parse_args()

    q = args.query
    if not q:
        q = input("Enter a question: ").strip()

    if q:
        if args.agent:
            run_agent_pipeline(q, max_steps=args.max_steps)
        else:
            run_rag_pipeline(q)
    else:
        print("No question entered.")
