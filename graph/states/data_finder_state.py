# Types
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field


def take_last_value(_old_value, new_value):
    return new_value


class InputState(TypedDict):
    path_to_json: Annotated[str, take_last_value]
    path_to_pdf: Annotated[str, take_last_value]  # Job description text


class OutputState(TypedDict):
    result: Annotated[dict, take_last_value]


# General State


class State(TypedDict):
    path_to_json: Annotated[str, take_last_value]
    path_to_pdf: Annotated[str, take_last_value]
    parameters: Annotated[list, take_last_value] 
    converted_pdf: Annotated[list, take_last_value]
    candidates: Annotated[list, take_last_value]
    param_page_map: Annotated[dict, take_last_value]
    result: Annotated[dict, take_last_value]

