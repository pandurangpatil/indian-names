#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract names from Maharashtra voters list PDFs in Devanagari script using spatial parsing.
Output format: Name (नाव), Husband/Father Name (पतीचे नाव / वडिलांचे नाव)
"""

import re
import argparse
import pdfplumber
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import pytesseract
import sys
from datetime import datetime


# OCR error correction dictionary
OCR_CORRECTIONS = {
    'नाच': 'नाव',
    'नाब': 'नाव',
    'यंडिलांचे': 'वडिलांचे',
    'यडिलांचे': 'वडिलांचे',
    'चडिलांचे': 'वडिलांचे',
    'बडिलांचे': 'वडिलांचे',
    'पत्तीचे': 'पतीचे',
    'पततीचे': 'पतीचे',
    'आणणाप्या': 'आण्णाप्पा',
    'आणिणांच्या': 'आण्णाच्या',
    'नांव': 'नाव',
    'ara:': 'नाव :',
    'aa:': 'नाव :',
    'att': '',  # OCR error, often appears as noise
    'Fart': '',  # OCR error for unknown text
    'Fert:': 'लिंग :',
    'fem:': 'लिंग :',
    'feat:': 'लिंग :',
    'fear': 'महिला',
    'mare': 'महिला',
    'ag:': 'वय :',
    'चय :': 'वय :',
    'बय :': 'वय :',
    'चे नाव:': 'वडिलांचे नाव:',
}


class ErrorLogger:
    """Logger for tracking extraction errors."""

    def __init__(self, output_path: str):
        self.errors = []
        self.output_path = output_path

    def log_error(self, pdf_file: str, page_num: int, raw_text: str, reason: str):
        """Log an extraction error."""
        self.errors.append({
            'pdf_file': pdf_file,
            'page_number': page_num,
            'raw_text': raw_text[:200],  # Limit text length
            'reason': reason
        })

    def save(self):
        """Save errors to file."""
        if self.errors:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EXTRACTION ERRORS LOG\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                for idx, error in enumerate(self.errors, 1):
                    f.write(f"\n--- Error #{idx} ---\n")
                    f.write(f"File: {error['pdf_file']}\n")
                    f.write(f"Page: {error['page_number']}\n")
                    f.write(f"Reason: {error['reason']}\n")
                    f.write(f"Raw Text:\n{error['raw_text']}\n")
                    f.write("-" * 80 + "\n")

            print(f"  ⚠ Logged {len(self.errors)} errors to: {self.output_path}")


def apply_ocr_corrections(text: str) -> str:
    """Apply OCR error corrections to text."""
    corrected = text
    for wrong, correct in OCR_CORRECTIONS.items():
        corrected = corrected.replace(wrong, correct)
    return corrected


def is_label_only(text: str) -> bool:
    """Check if text is only a label and not a name."""
    label_patterns = [
        r'^(नाव|पतीचे|वडिलांचे|आईचे|पत्नीचे|बडिलांचे)\s*[:：]?\s*$',
        r'^(विधानसभा|मतदारसंघ|क्रमांक|सांगली|महाराष्ट्र|छायाचित्र).*',
    ]

    for pattern in label_patterns:
        if re.search(pattern, text.strip()):
            return True

    return False


def has_excessive_ocr_errors(text: str) -> bool:
    """Check if text has excessive OCR corruption."""
    # Check for patterns indicating severe OCR issues
    problematic_patterns = [
        r'^[a-zA-Z]{1,3}:',  # Short English prefix like "aa:", "ga:"
        r'[a-zA-Z]{4,}\s+[a-zA-Z]{4,}',  # Multiple English words together
        r'eae|BT|donee|HATTERAS|HSS|decor|Fife|Orde|Sea|Bett|Fretar|ord|ga|Sta',  # Known OCR artifacts
        r'^\d+\s+(Fert|fem|feat|लिंग)\s*:',  # Age/gender info mistaken as name
        r'^\d{2,3}\s+(Fart|Fert)',  # Number + OCR error
    ]

    for pattern in problematic_patterns:
        if re.search(pattern, text):
            return True

    return False


def extract_name_from_line(line: str, label_pattern: str) -> str:
    """
    Extract name from a line with given label pattern.

    Args:
        line: Text line containing name
        label_pattern: Pattern like 'नाव' or 'पतीचे नाव'

    Returns:
        Extracted name or empty string
    """
    # Apply OCR corrections first
    line = apply_ocr_corrections(line)

    # Remove label and extract name
    patterns = [
        rf'{label_pattern}\s*[:：]\s*(.+)',
        rf'{label_pattern}\s+(.+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, line, re.UNICODE)
        if match:
            name = match.group(1).strip()

            # Clean up the name
            name = re.sub(r'\s+', ' ', name)  # Normalize spaces
            name = re.sub(r'^[-\s:：]+', '', name)  # Remove leading punctuation
            name = re.sub(r'[;|]+.*$', '', name)  # Remove everything after ; or |

            # Skip if it's just a label or has severe OCR errors
            if is_label_only(name) or has_excessive_ocr_errors(name):
                return ""

            return name.strip()

    return ""


def extract_voter_from_text_block(text: str, pdf_file: str, page_num: int, error_logger: ErrorLogger, _retry: bool = False) -> List[Dict[str, str]]:
    """
    Extract voter name and relationship name from a text block.
    Handles cases where multiple voters appear on the same line (side-by-side in PDF).

    Args:
        text: Text block from a voter card
        pdf_file: PDF filename for error logging
        page_num: Page number for error logging
        error_logger: Error logger instance
        _retry: Internal flag to prevent infinite recursion

    Returns:
        List of extracted voter dictionaries
    """
    voters = []

    # Split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Find name lines and relationship lines
    name_lines = []
    relationship_lines = []

    for line in lines:
        # Skip lines that are clearly not names
        if any(skip in line for skip in ['घर क्रमाक', 'वय :', 'लिंग :', 'छायाचित्र', 'उपलब्ध', 'यादी भाग', 'दिनांकास', 'विधानसभा', 'विभाग क्रमांक']):
            continue

        # Skip standalone numbers or IDs
        if re.match(r'^[\dA-Z\s\-]+$', line.strip()) and len(line.strip()) < 20:
            continue

        # Check if this line contains voter names
        if ('नाव' in line or 'नाब' in line) and (':' in line or ';' in line):
            # Check if it's the main name line (not relationship name line)
            if any(rel in line for rel in ['पतीचे', 'पत्नीचे', 'वडिलांचे', 'आईचे', 'बडिलांचे', 'पत्तीचे', 'चे नाव', 'चडिलांचे']):
                # This is a relationship name line
                relationship_lines.append(line)
            else:
                # This is a voter name line
                name_lines.append(line)

    # Handle case where multiple voters are on one line (side-by-side cards)
    # Pattern: "नाव : Person1 नाव ; Person2 नाव : Person3"
    all_voter_names = []
    all_relationship_names = []

    for name_line in name_lines:
        # Split by "नाव" occurrences to separate multiple voters
        # Use lookahead to keep "नाव" in the results
        parts = re.split(r'(?=नाव\s*[:;])', name_line)

        for part in parts:
            if 'नाव' in part:
                name = extract_name_from_line(part, 'नाव')
                if not name and 'नाब' in part:
                    name = extract_name_from_line(part, 'नाब')

                if name and not is_label_only(name) and not has_excessive_ocr_errors(name):
                    all_voter_names.append(name)

    for rel_line in relationship_lines:
        # Split by relationship patterns to handle multiple relationships on one line
        # Pattern: "पतीचे नाव: Name1 वडिलांचे नाव: Name2 पतीचे नाव: Name3"

        # Find all relationship patterns and their positions
        rel_matches = []
        for rel_pattern in ['पतीचे नाव', 'वडिलांचे नाव', 'पत्नीचे नाव', 'आईचे नाव', 'पत्तीचे नाव', 'चडिलांचे नाव', 'चे नाव']:
            for match in re.finditer(rf'{rel_pattern}\s*[:;]?\s*', rel_line):
                rel_matches.append((match.start(), match.end(), rel_pattern))

        # Sort by position
        rel_matches.sort(key=lambda x: x[0])

        # Extract each relationship name
        for i, (start, end, pattern) in enumerate(rel_matches):
            # Get text until next pattern or end of line
            if i < len(rel_matches) - 1:
                next_start = rel_matches[i + 1][0]
                text_part = rel_line[end:next_start]
            else:
                text_part = rel_line[end:]

            # Clean and extract the name
            rel_name = text_part.strip()
            # Remove any trailing patterns
            rel_name = re.sub(r'\s+(छायाचित्र|घर क्रमांक|उपलब्ध).*$', '', rel_name)
            rel_name = re.sub(r'\s+', ' ', rel_name).strip()

            if rel_name and len(rel_name) > 2 and not is_label_only(rel_name) and not has_excessive_ocr_errors(rel_name):
                all_relationship_names.append(rel_name)

    # Match voters with their relationship names
    # Typically, they appear in the same order on consecutive lines
    for i in range(len(all_voter_names)):
        voter_name = all_voter_names[i]
        relationship_name = all_relationship_names[i] if i < len(all_relationship_names) else ""

        # Clean up names
        voter_name = re.sub(r'^\s*(नाव|नाब)\s*[:;]\s*', '', voter_name)
        voter_name = re.sub(r'\s+', ' ', voter_name).strip()
        # Remove OCR artifacts
        voter_name = re.sub(r'\s*(att|Fart|Fert)\s*', ' ', voter_name).strip()
        voter_name = re.sub(r'\s+', ' ', voter_name).strip()

        if relationship_name:
            relationship_name = re.sub(r'^\s*(नाव|नाब)\s*[:;]\s*', '', relationship_name)
            relationship_name = re.sub(r'\s+', ' ', relationship_name).strip()
            # Remove OCR artifacts
            relationship_name = re.sub(r'\s*(att|Fart|Fert)\s*', ' ', relationship_name).strip()
            relationship_name = re.sub(r'\s+', ' ', relationship_name).strip()
            # Remove prefixes like 'ः' (colon in Marathi) or other punctuation
            relationship_name = re.sub(r'^[:\s;]+', '', relationship_name).strip()

        if len(voter_name) > 2 and not re.match(r'^\d+\s*$', voter_name):  # Not just numbers
            voters.append({
                'name': voter_name,
                'relationship_name': relationship_name
            })

    # Log errors for problematic extractions
    if not voters and text and 'नाव' in text and not _retry:
        # Only try correction once to avoid recursion
        corrected_text = apply_ocr_corrections(text)
        if corrected_text != text:
            return extract_voter_from_text_block(corrected_text, pdf_file, page_num, error_logger, _retry=True)
        else:
            # Only log if there was 'नाव' in the text but we couldn't extract
            error_logger.log_error(pdf_file, page_num, text, "Could not extract voter name from text block")

    return voters


def detect_voter_card_boxes(page) -> List[Tuple[float, float, float, float]]:
    """
    Detect individual voter card bounding boxes on a page using spatial analysis.

    Args:
        page: pdfplumber page object

    Returns:
        List of bounding boxes (x0, y0, x1, y1)
    """
    # Get all rectangles and lines from the page
    rects = page.rects
    lines = page.lines

    # Electoral roll pages typically have a grid layout with boxes for each voter
    # We'll detect boxes by looking for rectangular regions

    boxes = []

    if rects:
        # Use rectangles as voter card boundaries
        for rect in rects:
            x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
            width = x1 - x0
            height = y1 - y0

            # Filter boxes by size (voter cards are typically medium-sized)
            if width > 150 and height > 80 and width < 400 and height < 300:
                boxes.append((x0, y0, x1, y1))

    # If no rectangles found, try to detect boxes from lines
    if not boxes and lines:
        # Group lines into potential boxes
        # This is a simplified approach - a full implementation would be more sophisticated
        horizontal_lines = [l for l in lines if abs(l['height']) < 2]
        vertical_lines = [l for l in lines if abs(l['width']) < 2]

        # Sort lines
        horizontal_lines.sort(key=lambda l: l['top'])
        vertical_lines.sort(key=lambda l: l['x0'])

        # Create grid of potential boxes
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            for i in range(len(horizontal_lines) - 1):
                for j in range(len(vertical_lines) - 1):
                    x0 = vertical_lines[j]['x0']
                    y0 = horizontal_lines[i]['top']
                    x1 = vertical_lines[j + 1]['x0']
                    y1 = horizontal_lines[i + 1]['top']

                    width = x1 - x0
                    height = y1 - y0

                    if width > 150 and height > 80:
                        boxes.append((x0, y0, x1, y1))

    return boxes


def extract_voters_from_page(page, page_num: int, pdf_file: str, error_logger: ErrorLogger) -> List[Dict[str, str]]:
    """
    Extract voter information from a PDF page using OCR and text parsing.

    Args:
        page: pdfplumber page object
        page_num: Page number
        pdf_file: PDF filename for error logging
        error_logger: Error logger instance

    Returns:
        List of voter dictionaries
    """
    voters = []

    # Try text extraction first
    text = page.extract_text()

    # If no text or minimal text, use OCR
    if not text or len(text.strip()) < 100:
        try:
            img = page.to_image(resolution=300)
            pil_img = img.original
            text = pytesseract.image_to_string(pil_img, lang='mar+eng')
        except Exception as e:
            error_logger.log_error(pdf_file, page_num, "", f"OCR failed: {str(e)}")
            return voters

    if not text:
        return voters

    # Apply OCR corrections to entire text
    text = apply_ocr_corrections(text)

    # Split text into individual voter entries
    # Electoral rolls typically have serial numbers before each voter
    # Pattern: look for lines with just numbers, or numbers followed by name label

    # Split on voter serial numbers (lines starting with digits)
    lines = text.split('\n')

    current_block = []
    blocks = []

    for line in lines:
        line_stripped = line.strip()

        # Check if this is likely a voter serial number (standalone number or number at start)
        is_serial_number = False
        if line_stripped and line_stripped[0].isdigit():
            # Check if it's mostly digits
            # Serial numbers are typically 1-3 digits alone on a line
            if len(line_stripped) <= 3 and line_stripped.isdigit():
                is_serial_number = True
            elif re.match(r'^\d+\s*$', line_stripped):
                is_serial_number = True

        if is_serial_number and current_block:
            # Start of new voter entry, save previous block
            blocks.append('\n'.join(current_block))
            current_block = [line]
        else:
            current_block.append(line)

    # Don't forget the last block
    if current_block:
        blocks.append('\n'.join(current_block))

    # Process each block
    for block in blocks:
        if block.strip() and 'नाव' in block:
            block_voters = extract_voter_from_text_block(block, pdf_file, page_num, error_logger)
            voters.extend(block_voters)

    return voters


def extract_voters_from_pdf(pdf_path: str, start_page: int, end_page: int, error_logger: ErrorLogger) -> pd.DataFrame:
    """
    Extract voter names from PDF file using spatial parsing.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)
        error_logger: Error logger instance

    Returns:
        pandas DataFrame with extracted voter information
    """
    all_voters = []
    pdf_file = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        actual_end = min(end_page, total_pages)

        for page_num in range(start_page - 1, actual_end):
            print(f"    Page {page_num + 1}/{actual_end}...", end=" ")
            page = pdf.pages[page_num]

            voters = extract_voters_from_page(page, page_num + 1, pdf_file, error_logger)
            all_voters.extend(voters)
            print(f"{len(voters)} names")

    return pd.DataFrame(all_voters)


def process_folder(folder_path: Path, start_page: int, end_page: int, output_file: str) -> None:
    """
    Process all PDF files in a folder and extract voter names.

    Args:
        folder_path: Path to folder containing PDF files
        start_page: Starting page number for extraction
        end_page: Ending page number for extraction
        output_file: Output CSV filename
    """
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"✗ No PDF files found in: {folder_path}")
        return

    print("=" * 80)
    print(f"Maharashtra Voter Name Extractor (Spatial Parsing)")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Output format: Name, Husband/Father Name")
    print("=" * 80)

    # Initialize error logger
    error_file = output_file.replace('.csv', '_errors.txt')
    error_logger = ErrorLogger(error_file)

    all_names = []
    successful = 0
    failed = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        try:
            df = extract_voters_from_pdf(str(pdf_file), start_page, end_page, error_logger)

            if not df.empty:
                all_names.append(df)
                print(f"  ✓ Extracted {len(df)} names")
                successful += 1
            else:
                print(f"  ⚠ No names extracted")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            error_logger.log_error(pdf_file.name, 0, "", f"PDF processing failed: {str(e)}")
            failed += 1
            continue

    # Save error log
    error_logger.save()

    # Combine all results
    if all_names:
        final_df = pd.concat(all_names, ignore_index=True)

        # Remove duplicates
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates()
        duplicates_removed = initial_count - len(final_df)

        # Save to CSV with new column names
        final_df.columns = ['Name (नाव)', 'Husband/Father Name (पतीचे नाव / वडिलांचे नाव)']
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"✓ Total PDFs processed: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"✓ Total names extracted: {initial_count}")
        print(f"✓ Duplicates removed: {duplicates_removed}")
        print(f"✓ Unique names: {len(final_df)}")
        print(f"✓ Output saved to: {output_file}")
        print("=" * 80)

        # Display sample
        print("\nSample of extracted names (first 10):")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.max_colwidth', 50)
        print(final_df.head(10).to_string(index=False))
        print("-" * 80)

    else:
        print("\n✗ No names were extracted from any PDF files.")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract voter names from Maharashtra voters list PDFs (Devanagari script)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --folder /path/to/pdfs
  %(prog)s -f . --output names.csv
  %(prog)s -f ./pdfs --pages 3-32 --output all_names.csv
        """
    )

    parser.add_argument(
        '-f', '--folder',
        type=str,
        required=True,
        help='Path to folder containing PDF files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: extracted_names_<timestamp>.csv)'
    )

    parser.add_argument(
        '-p', '--pages',
        type=str,
        default='3-32',
        help='Page range to extract (default: 3-32). Format: START-END'
    )

    args = parser.parse_args()

    # Parse page range
    try:
        start_page, end_page = map(int, args.pages.split('-'))
        if start_page < 1 or end_page < start_page:
            raise ValueError
    except:
        print("✗ Error: Invalid page range. Use format: START-END (e.g., 3-32)")
        sys.exit(1)

    # Set output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"extracted_names_{timestamp}.csv"

    # Get folder path
    folder_path = Path(args.folder).resolve()
    if not folder_path.exists():
        print(f"✗ Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"✗ Error: Path is not a directory: {folder_path}")
        sys.exit(1)

    # Process folder
    process_folder(folder_path, start_page, end_page, output_file)


if __name__ == "__main__":
    main()
