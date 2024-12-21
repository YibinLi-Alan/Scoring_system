from docx import Document
import os

def extract_paragraphs(docx_file, output_file):
    try:
        # Verify file existence
        if not os.path.exists(docx_file):
            print(f"File not found: {docx_file}")
            return
        
        # Load the .docx file
        doc = Document(docx_file)
        
        # Extract paragraphs
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Save paragraphs to a new file in the desired format
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(",\n    ".join(f'"{paragraph}"' for paragraph in paragraphs))

        print(f"Extracted paragraphs saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "/Users/bin/Desktop/wmt/test.docx"  # Corrected file path
output_file = "/Users/bin/Desktop/wmt /extracted_paragraphs.txt"  # Output file
extract_paragraphs(input_file, output_file)
