# Standard
import json
import re

# Third-party
import pytesseract
from pdf2image import convert_from_path

# LangChain / LangGraph
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

# Project
from graph.states.data_finder_state import InputState, State, OutputState
from graph.utils.convert_to_model import convert_to_model 
from graph.utils.relevant_pages_model import convert_to_page_model
from graph.utils.variables import POPLER_PATH, DEFAULT_EXTRACTION_INSTRUCTIONS
from graph.utils.config import get_agent


def load_data(input_state: InputState) -> State:
    print("Loading parameters from JSON...")
    with open(input_state["path_to_json"], "r", encoding="utf-8") as f:
        parameters = json.load(f)

    return {
        "path_to_json": input_state["path_to_json"],
        "path_to_pdf": input_state["path_to_pdf"],
        "parameters": parameters
    }



def convert_pdf_text_pages(state: State) -> State:
    path_to_pdf = state["path_to_pdf"]

    # Step 1: Convert PDF pages to images
    images = convert_from_path(path_to_pdf, poppler_path=POPLER_PATH)

    # Step 2: Extract text from each image using pytesseract
    result = {}
    for i, image in enumerate(images):
        data = pytesseract.image_to_data(image, lang='heb+eng', output_type=pytesseract.Output.DICT)
        text = " ".join([word for word in data["text"] if word.strip() != ""])
        result[str(i + 1)] = text


    print(f"Loaded and OCR-scanned {len(result)} pages")

    return {
        "converted_pdf": result
    }



def find_relevant_pages(state: State) -> State:
    print("Running find_relevant_pages (batched)...")

    pages = state["converted_pdf"]
    parameters = state["parameters"]

    page_numbers = sorted([int(k) for k in pages.keys()])
    response_model = convert_to_page_model(parameters)
    agent = get_agent(response_model)

    combined_result = {param: {"pages": [], "summary": ""} for param in parameters}
    chunk_size = 10

    for i in range(0, len(page_numbers), chunk_size):
        chunk = page_numbers[i:i+chunk_size]
        chunk_text = "\n\n".join([
            f"Page {page_num}:\n{pages[str(page_num)]}"
            for page_num in chunk if str(page_num) in pages
        ])

        instructions = f"""
אתה מקבל טקסט של מכרז מתוך העמודים הבאים: {chunk}

המטרה שלך היא לזהות אילו עמודים רלוונטיים לכל אחד מהפרמטרים הבאים:
{', '.join(parameters)}

עבור כל פרמטר:
- אם עמודים שונים מכילים מידע זהה (למשל חזרה על שם הלקוח בלבד בלי מידע חדש), החזר רק את העמוד שבו זה הופיע לראשונה או שבו ההקשר הוא המשמעותי ביותר
- החזר רשימת מספרי עמודים (כפי שמופיעים בתחתית העמוד)
- תן הסבר קצר מה יש בכל עמוד שמצדיק את הקשר לפרמטר
- אל תכלול עמודים שמכילים רק מידע שחוזר על עצמו כמו שם המכרז או שם הלקוח בכותרת או בפסקת הפתיחה — אלא אם יש בהם מידע חדש או ייחודי
- אל תחזור על עמודים שכבר זוהו כאילו מכילים את אותו מידע בדיוק באצוות קודמות

החזר תשובה כ-JSON בלבד לפי הפורמט המוגדר מראש.
"""


        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": chunk_text}
        ]

        try:
            response = agent.invoke({"messages": messages})
            structured = response.get("structured_response", response)
            result = structured.model_dump()

            for param, data in result.items():
                combined_result[param]["pages"] += data["pages"]
                combined_result[param]["summary"] += " " + data["summary"]

        except Exception as e:
            print(f"Error in batch {chunk}: {e}")
            continue

    # Remove duplicate pages
    for param in combined_result:
        combined_result[param]["pages"] = sorted(list(set(combined_result[param]["pages"])))

    print("\n=== Final Relevant Pages Map ===")
    for param, data in combined_result.items():
        print(f"{param}: pages {data['pages']}, summary: {data['summary']}")

    return {
        "param_page_map": combined_result
    }



def info_extraction(state: State) -> State:
    
    print("\n Running info_extraction...")

    param_page_map = state["param_page_map"]

    # formatting the map to readable string
    page_map_summary = "\n".join([
    f"- {param}: pages {data['pages']}, summary: {data['summary']}"
    for param, data in param_page_map.items()
    ])
    

    instructions = DEFAULT_EXTRACTION_INSTRUCTIONS.format(page_map=page_map_summary)


    parameters = state["parameters"]
    pages = state["converted_pdf"]
    param_page_map = state["param_page_map"]

    response_model = convert_to_model(parameters)
    agent = get_agent(response_model)

    result = []

    print("Starting extraction per parameter...")

    #combining text per parameter
    for param in parameters:
        relevant_pages = param_page_map.get(param, {}).get("pages", [])
        print(f"\n Param: {param}, Pages: {relevant_pages}")

        combined_text = "\n\n".join([
            f"Page {page_num}:\n{pages[str(page_num)]}"
            for page_num in relevant_pages if str(page_num) in pages
        ])


        if not combined_text:
            print(f"No text found for parameter '{param}' – skipping")
            continue

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"parameters: ['{param}']\ntext:\n{combined_text}"}
        ]

        try:
            response = agent.invoke({"messages": messages})
            structured = response.get("structured_response", response)
            result.append(structured)

            value = structured.model_dump().get(param, {})
            print(value)


        except Exception as e:
            print(f"Error extracting {param}: {e}")
            continue

    print("Extraction completed.")

    return {
        "candidates": result,
    }




def classification(state: State) -> OutputState:
    combined_result = {}

    for candidate in state["candidates"]:
        result = candidate.result if hasattr(candidate, "result") else candidate
        for param, value in result.model_dump().items():
            new_score = value.get("score", 0)
            current = combined_result.get(param)

            if (
                current is None
                or new_score > current.get("score", 0)
                or (
                    new_score == current.get("score", 0)
                    and len(value.get("answer", "")) > len(current.get("answer", ""))
                )
            ):
                combined_result[param] = value

    return {"result": combined_result}
