from typing import List, Dict, Type
from pydantic import BaseModel, Field, create_model

class PageMatch(BaseModel):
    pages: List[int] = Field(..., description="List of page numbers (as printed in the PDF)")
    summary: str = Field(..., description="Short explanation of what appears on those pages")

def convert_to_page_model(parameters: List[str]) -> Type[BaseModel]:
    fields = {
        param: (PageMatch, Field(..., description=f"Pages relevant to {param}"))
        for param in parameters
    }

    return create_model("RelevantPagesModel", **fields)
