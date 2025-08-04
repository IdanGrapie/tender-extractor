from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent


MODEL_NAME = "gpt-4.1-micro"
TEMPERATURE = 0.2

def get_engine():
    return init_chat_model(model=MODEL_NAME, temperature=TEMPERATURE)

def get_agent(response_model):
    engine = get_engine()
    return create_react_agent(
        model=engine,
        response_format=response_model,
        tools=[]
    )
