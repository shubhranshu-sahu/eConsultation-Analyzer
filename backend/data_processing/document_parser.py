import fitz  # PyMuPDF
import re

class DocumentParser:
    """
    A robust and dynamic parser for official MCA notices. It relies on document
    structure (headings like 'Section X', 'Rule Y', '1. (1)') to be
    content-agnostic and handle a variety of legislative documents.
    """

    def _extract_clean_text(self, pdf_file_stream) -> str:
        """Extracts all text from a text-based PDF."""
        # This can be enhanced with the bilingual/OCR logic we discussed
        with fitz.open(stream=pdf_file_stream.read(), filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        return full_text

    def parse(self, pdf_file_stream) -> dict:
        """
        Parses a PDF into a structured dictionary of sections.
        """
        structured_content = {}
        try:
            full_text = self._extract_clean_text(pdf_file_stream)
            
            # This robust regex finds various legal heading formats.
            # It looks for patterns at the start of a line to be more accurate.
            pattern = re.compile(
                r'^\s*('
                r'Section\s+\d+[A-Z]?\s*(?:\(\d+\))?|'  # Matches "Section 233", "Section 1(1)"
                r'Rule\s+\d+[A-Z]?\s*(?:\(\d+\))?|'     # Matches "Rule 25", "Rule 11(2)"
                r'\d+\.\s*\(\d+\)|'                     # Matches "1. (1)"
                r'(?:SCHEDULE-[IVX]+)'                 # Matches "SCHEDULE-I"
                r')\s*[:.-]?', re.IGNORECASE | re.MULTILINE
            )
            
            matches = list(pattern.finditer(full_text))
            
            if not matches:
                structured_content["Document Level"] = full_text.strip().replace('\n', ' ')
                return structured_content

            # Capture the Preamble (text before the first structural heading)
            first_match_start = matches[0].start()
            preamble = full_text[:first_match_start].strip().replace('\n', ' ')
            if preamble:
                structured_content["Preamble / Background"] = preamble

            # Dynamically extract content between headings
            for i, current_match in enumerate(matches):
                # Clean up the heading to use as a key
                heading = re.sub(r'[\s.-]+$', '', current_match.group(1).strip())
                
                start_pos = current_match.end()
                end_pos = matches[i+1].start() if i + 1 < len(matches) else len(full_text)
                
                content = full_text[start_pos:end_pos].strip().replace('\n', ' ')
                
                structured_content[heading] = content
            
            return structured_content

        except Exception as e:
            print(f"Error parsing document: {e}")
            return {"error": "Failed to parse the document."}

# Example Usage:
if __name__ == '__main__':
    parser = DocumentParser()
    # List of files to test our universal parser on
    files_to_test =  [
    #"C:\\Users\\Vansh\\Desktop\\SIH\\notice docs\\Draft-Notification Public Notice inviting comments on draft notification for amending the Companies (Compromises, Arrangements and Amalgamations) Rules, 2016-reg.pdf",
    #"C:\\Users\\Vansh\\Desktop\\SIH\\notice docs\\draft-rr-notification-20250210 Inviting comments of all stakeholders on draft amended Recruitment Rules for the post of Prosecutor and Senior Prosecutor in the Serious Fraud Investigation Office (SFIO) under Ministry of Co.pdf",
   # "C:\\Users\\Vansh\\Desktop\\SIH\\notice docs\\Notice inviting comments on draft Companies (Compromises, Arrangements and Amalgamations) Amendment Rules, 2025.pdf",
    #"C:\\Users\\Vansh\\Desktop\\SIH\\notice docs\\notice-20250626 Notice inviting comments on draft Companies (Meetings of Board and its Powers) Amendment Rules, 2025.pdf",
    "Public-notice-bilingual-languge-20250721.pdf"
]
    
    for file_path in files_to_test:
        print(f"\n--- Parsing Document: {file_path.split('/')[-1]} ---")
        try:
            with open(file_path, "rb") as f:
                parsed_data = parser.parse(f)
                import json
                print(json.dumps(parsed_data, indent=2))
        except FileNotFoundError:
            print(f"ERROR: File not found at '{file_path}'. Please update the path.")