from graph.nodes.data_finder_nodes import load_data, convert_pdf_text_pages, regex_filter_pages
from langchain_community.document_loaders import PyPDFLoader


# Provide your file paths here
state = load_data({
    "path_to_json": "parameters_for_exercise.json",
    "path_to_pdf":  "mei_sheva.pdf"
})


state = convert_pdf_text_pages(state)
state = regex_filter_pages(state)

# Pretty print
from pprint import pprint
pprint(state["param_page_map"])

'''
def load_data(input_state: InputState) -> State:

    with open(input_state["path_to_json"], "r") as f:
        data = json.load(f)
    return {
        "path_to_json": input_state["path_to_json"],
        "parameters" : data,
        "path_to_pdf": input_state["path_to_pdf"]
    }
        response = agent.invoke({"messages": messages})
        result.append(response.get("structured_response", response))'''

'''
        try:
            response = agent.invoke({"messages": messages})
            five_score_flag[] # make true if score == 5

            result.append(response.get("structured_response", response))
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            continue '''