from typing import List, Dict, Type
from pydantic import BaseModel, Field, create_model

class SubModel(BaseModel):
    answer: str = Field(default="", description="The extracted answer")
    details: str = Field(default="", description="Extra context or explanation")
    source: str = Field(default="", description="Page + Where the answer was found")
    score: int = Field(default=0, ge=0, le=5, description="Confidence score from 0 (not found) to 5 (high confidence)")

def convert_to_model(parameters: List[str]) -> Type[BaseModel]:
    fields = {
        param: (SubModel, Field(..., description=f"{param} extraction result"))
        for param in parameters
    }

    return create_model("DynamicModel", **fields)
