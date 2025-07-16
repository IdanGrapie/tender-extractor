import sys
from graph.data_finder_flow import data_finder_flow
from dotenv import load_dotenv
import json
import os

# Load environment variables from .env file
load_dotenv()

def help():
    return """to use for pass json and pdf"""

def main():

    path_to_json = sys.argv[1]
    path_to_pdf = sys.argv[2]

    if path_to_json in ["--help", "--h"] or not path_to_pdf:
        return help()
    

    # Run the data extraction flow
    result = data_finder_flow.invoke({
        "path_to_json": path_to_json,
        "path_to_pdf": path_to_pdf

    })
    print(result)

    # Generate output file name based on PDF name
    pdf_name = os.path.splitext(os.path.basename(path_to_pdf))[0]
    output_path = f"{pdf_name}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved result to {output_path}")
        

if __name__ == "__main__":
    main()
