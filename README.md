# Tender Extractor

AI-powered document extraction pipeline for Hebrew tender PDF documents.

## Overview

Tender Extractor is a Python-based project that extracts structured information from Hebrew tender documents.

The system receives a PDF tender document and a JSON file that defines the required extraction fields. It then analyzes the document and returns a structured JSON output containing the extracted answers, supporting details, source references, and confidence scores.

This project focuses on turning long, unstructured PDF documents into clean, machine-readable data.

## Problem

Tender documents are often long, complex, and difficult to review manually. Important information such as the client name, tender name, threshold conditions, contract period, evaluation method, and bid guarantee may be spread across many pages.

This project automates the extraction process using an LLM-based workflow, making it easier to identify key information from tender documents in a consistent and structured way.

## Extracted Fields

The system can extract fields such as:

* Client name
* Tender name
* Threshold conditions
* Contract period
* Evaluation method
* Bid guarantee

The extraction parameters are defined in a JSON file, so the workflow can be adjusted for different document types or business requirements.

## Pipeline

```text
PDF → page text extraction → relevant page detection → LLM extraction → structured JSON → confidence scoring
```

## How It Works

1. The user provides a PDF document and a JSON file with the requested extraction fields.
2. The system reads the PDF and extracts text from the document.
3. The workflow searches for the most relevant content for each requested parameter.
4. The relevant context is sent to an LLM for structured information extraction.
5. The result is returned as JSON.
6. Each extracted field includes an answer, supporting details, source information, and a confidence score.

## Example Output

```json
{
  "client_name": {
    "answer": "מי שבע - תאגיד אזורי למים וביוב בע\"מ",
    "details": "The client name appears clearly in the tender document.",
    "source": "Page 2, main heading",
    "score": 5
  },
  "bid_guarantee": {
    "answer": "10,000 ₪",
    "details": "The bid guarantee amount is explicitly mentioned in the tender requirements.",
    "source": "Tender requirements section",
    "score": 5
  }
}
```

## Confidence Score

Each extracted field receives a score from 0 to 5:

| Score | Meaning                                            |
| ----- | -------------------------------------------------- |
| 5     | Clear answer found with strong supporting evidence |
| 3-4   | Partial or less direct evidence found              |
| 1-2   | Weak or uncertain evidence                         |
| 0     | The answer was not found in the document           |

If the information does not appear in the document, the system should return `לא נמצא` instead of guessing.

## Tech Stack

* Python
* LLM-based extraction
* Prompt engineering
* JSON structured output
* PDF text extraction
* Confidence scoring
* Document automation

## Project Structure

```text
tender-extractor/
├── main.py
├── requirements.txt
├── parameters_for_exercise.json
├── README.md
├── graph/
│   └── data_finder_flow.py
├── examples/
│   ├── input/
│   │   └── sample_tender.pdf
│   └── output/
│       └── sample_result.json
└── .gitignore
```

> Note: The exact structure may change as the project develops.

## Setup

Clone the repository:

```bash
git clone https://github.com/IdanGrapie/tender-extractor.git
cd tender-extractor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

> Do not commit your `.env` file to GitHub.

## Usage

Run the extraction pipeline:

```bash
python main.py parameters_for_exercise.json mei_sheva.pdf
```

Example with an output file:

```bash
python main.py parameters_for_exercise.json mei_sheva.pdf --output mei_sheva_results.json
```

## Input Parameters Example

The extraction parameters are defined in a JSON file.

Example:

```json
{
  "parameters": [
    "client_name",
    "tender_name",
    "threshold_conditions",
    "contract_period",
    "evaluation_method",
    "bid_guarantee"
  ]
}
```

## Why This Project Matters

This project demonstrates practical AI automation skills that are useful in real business workflows:

* Extracting structured data from unstructured documents
* Working with Hebrew PDF documents
* Designing prompts for reliable information extraction
* Returning machine-readable JSON output
* Using source references to support extracted answers
* Adding confidence scoring instead of returning unverified results
* Building reusable workflows for document analysis

## Future Improvements

Planned improvements include:

* Add a Streamlit interface for uploading PDFs
* Add automated tests for extraction logic
* Add support for more document types
* Improve page-level source references
* Add evaluation against expected output files
* Add Docker support
* Add CI checks with GitHub Actions


