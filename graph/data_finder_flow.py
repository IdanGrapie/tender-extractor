from langgraph.graph import StateGraph, START, END
from graph.states.data_finder_state import InputState, OutputState, State
from graph.nodes.data_finder_nodes import load_data, convert_pdf_text_pages, classification, regex_filter_pages, scorer, recheck


# Flow
workflow = StateGraph(State, input=InputState, output=OutputState)

# Nodes
workflow.add_node("load_data", load_data)
workflow.add_node("convert_pdf_text_pages", convert_pdf_text_pages)
workflow.add_node("regex_filter_pages", regex_filter_pages)
workflow.add_node("scorer", scorer)
workflow.add_node("recheck", recheck)
workflow.add_node("classification", classification)


# Edges
workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "convert_pdf_text_pages")
workflow.add_edge("convert_pdf_text_pages", "regex_filter_pages")
workflow.add_edge("regex_filter_pages", "scorer")
workflow.add_conditional_edges("scorer", lambda x: "classification" if all(x["five_score_flag"].values()) else "recheck")
workflow.add_edge("recheck", "classification")
workflow.add_edge("classification", END)


# Compile
data_finder_flow = workflow.compile()
