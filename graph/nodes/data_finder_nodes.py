from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from graph.states.data_finder_state import InputState, State, OutputState
import json
from langchain.document_loaders import PyPDFLoader
from pydantic import create_model
from graph.utils.convert_to_model import convert_to_model 
import re


def load_data(input_state: InputState) -> State:
    
    with open(input_state["path_to_json"], "r", encoding="utf-8") as f:
        parameters = json.load(f)

    return {
        "path_to_json": input_state["path_to_json"],
        "path_to_pdf": input_state["path_to_pdf"],
        "parameters": parameters,
    }


def convert_pdf_text_pages(state: State) -> State:
    loader = PyPDFLoader(state["path_to_pdf"])
    pages = loader.load_and_split()

    result = [page.page_content for page in pages]

    return {"converted_pdf": result }



def regex_filter_pages(state: State) -> State:
    keywords = {
    "client_name": [r"בע[\"״]מ", r"אנחנו\s?ב", r"תאגיד"],
    "tender_name": [r"מספר\s?מכרז", r"מכרז\s?פומבי"],
    "threshold_conditions": [r"תנאי\s?סף", r"השתתפות.*?מותנית", r"דרישות.*?סף"],
    "contract_period": [r"תקופת\s?ההסכם", r"משך\s?ההסכם", r"תוקף.*?הסכם"],
    "evaluation_method": [r"אופן.*?בחירת.*?הזוכה", r"מדדים.*?לבחירה", r"מחיר"],
    "bid_guarantee": [r"ערבות.*?מכרז", r"דמי\s?השתתפות", r"סכום.*?שיש\s?להפקיד"],
    "idea_author": [r"רעיון.*?מקורי", r"הוגה.*?הרעיון", r"יזם"]
    }
    
    
    
    param_page_map = {param: [] for param in keywords}

    for i, page in enumerate(state["converted_pdf"]):
        for param, patterns in keywords.items():
            for pattern in patterns:
                if re.search(pattern, page):
                    param_page_map[param].append(i + 1)  # 1-based indexing
                    break

    return {
        "param_page_map": param_page_map
    }




def scorer(state: State) -> State:


    instructions = f"""
אתה מומחה באיתור נתונים     

    המטרה שלך היא לאתר ולחלץ ערכים מדויקים מתוך טקסט של מכרז לפי שמות פרמטרים מוגדרים מראש.

    אתה מצויד בטקסט ופרמטרים אותם תצטרך למצוא על גבי הטקסט.

    הוראות:
    - אם ערך מופיע יותר מפעם אחת — בחר את הנוסח הכי שלם, ברור ומדויק.
    -threshold_conditions: תחפש את תנאים המופעים בטופס ואתה רשאי לפרט 
    -contract_period: כמות הזמן שהחוזה יהיה בתוקף, נמצא תחת כותרות כגון: תקופת ההסכם, זמן ההסכם, משך החוזה 
    -bid_guarantee:תחפש את סכום הכסף שצריך לקבל
    -evaluation method: תחפש מתחת לכותרות כגון חישוב משוקלל, מרכיבי ההצעה, וחפש מילות מפתח כגון איכות, מחיר, זמן    
    - בשדה "source" ציין תמיד את מספר העמוד ואת המיקום בו נמצא המידע: פסקה, שורה או טבלה. לדוגמה: "עמוד 2, פסקה ראשונה".
    - ציין תמיד את מספר העמוד **כפי שהוא מופיע בתחתית הדף**, לא לפי המספור במערכת.
    

    חישוב score:
    - אם לא מצאת מידע על פרמטר מסוים, תחזיר עבורו:
    answer: ""
    details: ""
    source: "לא נמצא"
    score: 0
    - אם אתה חושב שהמידע יכול להיות קשור אבל לא בוודאות תחזיר 3
    - אם אתה בטוח שהמידע נכון תחזיר 5

    ---

    הפרמטרים לחילוץ יופיעו מיד לאחר ההוראות. לאחר מכן יופיע הטקסט המלא של המכרז ממנו יש לחלץ את הערכים.
    """


    response_model = convert_to_model(state["parameters"])


    engine = init_chat_model(
        model="gpt-4.1",
        temperature=0.2,
    )

    agent = create_react_agent(
        model=engine,
        response_format=response_model,
        tools=[] # use to trigger functions
    )

    result = []

    five_score_flag = {param: False for param in state["parameters"]}

    param_page_map = state.get("param_page_map", {}) 

    # Create a union of all pages used across all parameters
    used_pages = sorted(set(
        page
        for pages in param_page_map.values()
        for page in pages
    ))

    for page_num in used_pages:
        page = state["converted_pdf"][page_num - 1]


        messages = [
            {"role": "system", "content": instructions},  #sends both the extraction instructions and the current page text.
            {"role": "user", "content": f'parameters:{state["parameters"]} text:{page}'}
        ]


        try:
            response = agent.invoke({"messages": messages})
            structured = response.get("structured_response", response)

       
            for param, value in structured.model_dump().items():
                if value.get("score") == 5:
                    five_score_flag[param] = True

            result.append(structured)

        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            continue # Skip to the next page instead of failing the entire run
    return {
        "candidates": result,
        "five_score_flag": five_score_flag  # Used to check if any param is still missing after scoring
    }



def recheck(state: State) -> State:

    instructions = f"""
אתה מומחה באיתור נתונים     

    המטרה שלך היא לאתר ולחלץ ערכים מדויקים מתוך טקסט של מכרז לפי שמות פרמטרים מוגדרים מראש.

    אתה מצויד בטקסט ופרמטרים אותם תצטרך למצוא על גבי הטקסט.

    הוראות:
    - אם ערך מופיע יותר מפעם אחת — בחר את הנוסח הכי שלם, ברור ומדויק.
    -threshold_conditions: תחפש את תנאים המופעים בטופס
    -contract_period: כמות הזמן שהחוזה יהיה בתוקף, נמצא תחת כותרות כגון: תקופת ההסכם, זמן ההסכם, משך החוזה 
    -bid_guarantee:תחפש את סכום הכסף שצריך לקבל
    -evaluation method: תחפש מתחת לכותרות כגון חישוב משוקלל, מרכיבי ההצעה, וחפש מילות מפתח כגון איכות, מחיר, זמן    
    - בשדה "source" ציין תמיד את מספר העמוד ואת המיקום בו נמצא המידע: פסקה, שורה או טבלה. לדוגמה: "עמוד 2, פסקה ראשונה".
    - ציין תמיד את מספר העמוד **כפי שהוא מופיע בתחתית הדף**, לא לפי המספור במערכת.
    -

    חישוב score:
    - אם לא מצאת מידע על פרמטר מסוים, תחזיר עבורו:
    answer: ""
    details: ""
    source: "לא נמצא"
    score: 0
    - אם אתה חושב שהמידע יכול להיות קשור אבל לא בוודאות תחזיר 3
    - אם אתה בטוח שהמידע נכון תחזיר 5

    ---

    הפרמטרים לחילוץ יופיעו מיד לאחר ההוראות. לאחר מכן יופיע הטקסט המלא של המכרז ממנו יש לחלץ את הערכים.
    """
    engine = init_chat_model("gpt-4.1", temperature=0.2)
    response_model = convert_to_model(state["parameters"])
    agent = create_react_agent(
        model=engine,
        response_format=response_model,
        tools=[] # use to trigger functions
    )


    total_pages = len(state["converted_pdf"])
    # gives only the scanned pages
    used_pages = set(
        page
        for pages in state.get("param_page_map", {}).values()
        for page in pages
    )
    
    missed_pages = set(range(1, total_pages + 1)) - used_pages

    results = []

    for page_num, page in enumerate(state["converted_pdf"], start=1):
        if page_num not in missed_pages:
            continue

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f'parameters:{state["parameters"]} text:{page}'}
        ]

        try:
            response = agent.invoke({"messages": messages})
            structured = response.get("structured_response", response)
            results.append(structured)
        except Exception as e:
            print(f" Error on recheck page {page_num}: {e}")
            continue

    return {"candidates": results}



def classification(state: State) -> OutputState:
    
    combined_result = {}

    for candidate in state["candidates"]:
        result = candidate.result if hasattr(candidate, "result") else candidate
        for param, value in result.model_dump().items():
            if param not in combined_result or value.get("score", 0) > combined_result[param].get("score", 0):
                combined_result[param] = value

    
   
    return {"result": combined_result}





