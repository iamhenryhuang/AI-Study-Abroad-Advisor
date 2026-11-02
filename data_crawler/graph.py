"""LangGraph assembly.

Main graph (SchoolState, thread_id = school_id):
  init_crawl ─▶ discover_urls ─▶ filter_discovered_urls(batch LLM) ⟲
      └▶ plan_scrape ═Send═▶ scrape_page(Node 4) ─▶ collect_scraped (final-URL dedup)
      └▶ ═Send═▶ process_page (page subgraph) ─▶ sufficiency_evaluator(Node 9)
            └─ coverage report ─▶ tagging ─▶ chunking ─▶ embedding
                                                ─▶ db_writer ─▶ finalize_school

Page subgraph (ProcessState):
  content_classification(Node 5, LLM)
      ├─ irrelevant/faculty ─▶ discard_page
      └─ relevant ─▶ identify_programs(Node 6) ─▶ structured_extraction(Node 7)
                     ─▶ hallucination_validation(Node 8)
                          ─▶ semantic_repair (batch fix and fill-in from the full original text)
                          ─▶ re-validate ─▶ finalize_page (residual problems go to review_items)
"""
from langgraph.graph import StateGraph, START, END

from .state import SchoolState, ProcessState
from .nodes import page as np_
from .nodes import school as ns


def build_page_subgraph():
    g = StateGraph(ProcessState)
    g.add_node("content_classification", np_.content_classification)
    g.add_node("discard_page", np_.discard_page)
    g.add_node("identify_programs", np_.identify_programs)
    g.add_node("structured_extraction", np_.structured_extraction)
    g.add_node("hallucination_validation", np_.hallucination_validation)
    g.add_node("semantic_repair", np_.semantic_repair)
    g.add_node("finalize_page", np_.finalize_page)

    g.add_edge(START, "content_classification")
    g.add_conditional_edges("content_classification", np_.is_relevant_content,
                            ["identify_programs", "discard_page"])
    g.add_edge("discard_page", END)
    g.add_conditional_edges("identify_programs", np_.has_program_target,
                            ["structured_extraction", "finalize_page"])
    g.add_edge("structured_extraction", "hallucination_validation")
    g.add_conditional_edges("hallucination_validation", np_.extraction_quality_check,
                            ["semantic_repair", "finalize_page"])
    g.add_edge("semantic_repair", "hallucination_validation")
    g.add_edge("finalize_page", END)
    return g.compile()


def build_school_graph(checkpointer=None):
    page_subgraph = build_page_subgraph()

    def process_page(state: ProcessState) -> dict:
        """A wrapper: write only the accumulating keys back to the main graph, so parallel branches do not overwrite single-valued keys such as school_id at the same time."""
        result = page_subgraph.invoke(state)
        return {"page_results": result.get("page_results", []),
                "review_items": result.get("review_items", [])}

    g = StateGraph(SchoolState)
    g.add_node("init_crawl", ns.init_crawl)
    g.add_node("discover_urls", ns.discover_urls)
    g.add_node("filter_discovered_urls", ns.filter_discovered_urls)
    g.add_node("plan_scrape", ns.plan_scrape)
    g.add_node("scrape_page", ns.scrape_page)
    g.add_node("collect_scraped", ns.collect_scraped)
    g.add_node("process_page", process_page)
    g.add_node("sufficiency_evaluator", ns.sufficiency_evaluator)
    g.add_node("tagging", ns.tagging)
    g.add_node("chunking", ns.chunking)
    g.add_node("embedding", ns.embedding)
    g.add_node("db_writer", ns.db_writer)
    g.add_node("finalize_school", ns.finalize_school)

    g.add_edge(START, "init_crawl")
    g.add_conditional_edges("init_crawl", ns.after_init,
                            ["discover_urls", "finalize_school"])
    g.add_edge("discover_urls", "filter_discovered_urls")
    g.add_conditional_edges("filter_discovered_urls", ns.should_continue_bfs,
                            ["discover_urls", "plan_scrape"])
    g.add_conditional_edges("plan_scrape", ns.dispatch_scrape,
                            ["scrape_page", "collect_scraped"])
    g.add_edge("scrape_page", "collect_scraped")
    g.add_conditional_edges("collect_scraped", ns.dispatch_process,
                            ["process_page", "sufficiency_evaluator"])
    g.add_edge("process_page", "sufficiency_evaluator")
    g.add_edge("sufficiency_evaluator", "tagging")
    g.add_edge("tagging", "chunking")
    g.add_edge("chunking", "embedding")
    g.add_edge("embedding", "db_writer")
    g.add_edge("db_writer", "finalize_school")
    g.add_edge("finalize_school", END)

    return g.compile(checkpointer=checkpointer)
