"""Command-line entry point for ``python -m retriever.agent``."""

import argparse
import io
import sys

from .runtime import run_agent


def main() -> int:
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="Agentic RAG (LangGraph)")
    parser.add_argument("query", nargs="?", help="the user's question")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="safety valve (the actual recursion_limit = max_steps * 4)",
    )
    args = parser.parse_args()

    query = args.query or input("Enter a question: ").strip()
    if not query:
        print("No question entered.")
        return 1

    answer = run_agent(query, max_steps=args.max_steps, verbose=True)
    if not answer:
        print("Failed to generate an answer.")
        return 1

    print("\n" + "=" * 30 + " Final answer " + "=" * 30)
    print(answer)
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
