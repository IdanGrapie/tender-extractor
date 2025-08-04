from langgraph.graph import StateGraph, START, END
from graph.states.data_finder_state import InputState, OutputState, State
from graph.nodes.data_finder_nodes import load_data, convert_pdf_text_pages, classification, find_relevant_pages, info_extraction


# Flow
workflow = StateGraph(State, input=InputState, output=OutputState)

# Nodes
workflow.add_node("load_data", load_data)
workflow.add_node("convert_pdf_text_pages", convert_pdf_text_pages)
workflow.add_node("find_relevant_pages", find_relevant_pages)
workflow.add_node("info_extraction", info_extraction)
workflow.add_node("classification", classification)


# Edges
workflow.add_edge(START, "load_data")
workflow.add_edge("load_data", "convert_pdf_text_pages")
workflow.add_edge("convert_pdf_text_pages", "find_relevant_pages")
workflow.add_edge("find_relevant_pages", "info_extraction")
workflow.add_edge("info_extraction", "classification")
workflow.add_edge("classification", END)


# Compile
data_finder_flow = workflow.compile()
