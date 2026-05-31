import argparse
import json
import os

from dotenv import load_dotenv
from graph.data_finder_flow import data_finder_flow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract structured information from Hebrew tender PDF documents."
    )

    parser.add_argument(
        "path_to_json",
        help="Path to the JSON file that defines the extraction parameters."
    )

    parser.add_argument(
        "path_to_pdf",
        help="Path to the PDF document."
    )

    parser.add_argument(
        "--output",
        help="Optional path for the output JSON file."
    )

    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()

    result = data_finder_flow.invoke({
        "path_to_json": args.path_to_json,
        "path_to_pdf": args.path_to_pdf
    })

    pdf_name = os.path.splitext(os.path.basename(args.path_to_pdf))[0]
    output_path = args.output or f"{pdf_name}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Data extraction completed")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
