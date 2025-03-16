from dotenv import load_dotenv
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

load_dotenv()

parser = LlamaParse(
            result_type="markdown",
            premium_mode=True,
            api_key='llx-pEEtT2TPSKKbWOm95SuEZM1KhlQaElXJXP4E06IG9JerB23q'
        )  # Choose between "markdown" or "text"

# Define paths
pdf_file_path = "cnn1.pdf"
output_md_path = "md_output/output2.md"  # Ensure it's a valid path

# Ensure output directory exists
os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

# Load and parse PDF
try:
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[pdf_file_path], file_extractor=file_extractor).load_data()

    if not documents:
        raise ValueError("No data was parsed from the provided PDF.")

    # Extract Markdown text
    markdown_content = "\n\n".join([doc.text for doc in documents])  # Ensure doc has a 'text' attribute

    # Print output (optional)
    print(markdown_content)

    # Save Markdown content to file
    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Markdown saved to: {output_md_path}")

except Exception as e:
    raise ValueError(f"Error parsing PDF: {e}")
