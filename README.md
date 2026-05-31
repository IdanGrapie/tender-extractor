# Tender Extractor

AI-powered Hebrew tender document extraction pipeline.

## What it does
Extracts structured fields from Hebrew PDF tender documents into validated JSON.

## Extracted fields
- Client name
- Tender name
- Threshold conditions
- Contract period
- Evaluation method
- Bid guarantee

## Tech Stack
Python, LangGraph, LLMs, PyMuPDF, Pydantic, JSON validation

## Example Output
```json
{
  "client_name": {
    "answer": "מי שבע",
    "source_page": 3,
    "confidence": 5
  }
}
```

## Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/IdanGrapie/tender-extractor.git
cd tender-extractor
pip install -r requirements.txt
```

## Run
python3 main.py **path/to/parameters.json** **path/to/document.pdf**
