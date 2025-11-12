#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract names from Maharashtra voters list PDFs in Devanagari script using spatial parsing.
Output format: Name (नाव), Husband/Father Name (पतीचे नाव / वडिलांचे नाव)
"""

import re
import argparse
import pdfplumber
from pathlib import Path
from typing import List, Tuple
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
    """Logger for tracking extraction errors and rejected names."""

    def __init__(self, output_path: str):
        self.errors = []
        self.rejected_names = []
        self.output_path = output_path

    def log_error(self, pdf_file: str, page_num: int, raw_text: str, reason: str):
        """Log an extraction error."""
        self.errors.append({
            'pdf_file': pdf_file,
            'page_number': page_num,
            'raw_text': raw_text[:200],  # Limit text length
            'reason': reason
        })

    def log_rejected_name(self, pdf_file: str, page_num: int, name: str, reason: str):
        """Log a rejected name with reason."""
        self.rejected_names.append({
            'pdf_file': pdf_file,
            'page_number': page_num,
            'name': name,
            'reason': reason
        })

    def save(self):
        """Save errors and rejected names to file."""
        if self.errors or self.rejected_names:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EXTRACTION ERRORS AND REJECTED NAMES LOG\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                if self.errors:
                    f.write(f"EXTRACTION ERRORS ({len(self.errors)} total)\n")
                    f.write("=" * 80 + "\n")
                    for idx, error in enumerate(self.errors, 1):
                        f.write(f"\n--- Error #{idx} ---\n")
                        f.write(f"File: {error['pdf_file']}\n")
                        f.write(f"Page: {error['page_number']}\n")
                        f.write(f"Reason: {error['reason']}\n")
                        f.write(f"Raw Text:\n{error['raw_text']}\n")
                        f.write("-" * 80 + "\n")

                if self.rejected_names:
                    f.write(f"\n\nREJECTED NAMES ({len(self.rejected_names)} total)\n")
                    f.write("=" * 80 + "\n")
                    for idx, rejected in enumerate(self.rejected_names, 1):
                        f.write(f"\n--- Rejected #{idx} ---\n")
                        f.write(f"File: {rejected['pdf_file']}\n")
                        f.write(f"Page: {rejected['page_number']}\n")
                        f.write(f"Name: {rejected['name']}\n")
                        f.write(f"Reason: {rejected['reason']}\n")
                        f.write("-" * 80 + "\n")

            total = len(self.errors) + len(self.rejected_names)
            print(f"  ⚠ Logged {total} issues to: {self.output_path}")


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


def is_valid_devanagari_name(name: str) -> Tuple[bool, str]:
    """
    Strictly validate if text is a valid Devanagari name.

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    name = name.strip()

    # Check minimum length
    if len(name) < 3:
        return False, "Too short (less than 3 characters)"

    # Check if it's only a label
    if is_label_only(name):
        return False, "Label only (not a name)"

    # Check for English letters
    if re.search(r'[a-zA-Z]', name):
        return False, "Contains English characters"

    # Check for numbers (except in rare cases where numbers might be part of name)
    if re.search(r'\d', name):
        return False, "Contains numbers"

    # Check for special characters (allow only Devanagari and basic punctuation)
    # Devanagari range: U+0900–U+097F
    # Allow: space, apostrophe ('), hyphen (-), comma (,)
    allowed_pattern = r'^[\u0900-\u097F\s\'\-,]+$'
    if not re.match(allowed_pattern, name):
        return False, "Contains invalid special characters"

    # Check for known OCR artifacts
    ocr_artifacts = ['att', 'eae', 'BT', 'donee', 'HATTERAS', 'HSS', 'decor',
                     'Fife', 'Orde', 'Sea', 'Bett', 'Fretar', 'ord', 'ga', 'Sta',
                     'Fart', 'Fert', 'fem', 'feat', 'fear', 'mare', 'ara', 'aa']
    for artifact in ocr_artifacts:
        if artifact in name:
            return False, f"OCR artifact detected: '{artifact}'"

    # Check if it's mostly punctuation
    if len(re.findall(r'[\u0900-\u097F]', name)) < 3:
        return False, "Insufficient Devanagari characters"

    # Check for excessive punctuation
    punctuation_count = len(re.findall(r'[\'\-,;:|]', name))
    char_count = len(name.replace(' ', ''))
    if char_count > 0 and punctuation_count / char_count > 0.3:
        return False, "Excessive punctuation"

    return True, ""


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


def extract_voter_from_text_block(text: str, pdf_file: str, page_num: int, error_logger: ErrorLogger, _retry: bool = False) -> List[str]:
    """
    Extract ALL names from a text block (voters, husbands, fathers, mothers, wives).
    Each name is returned as a separate entry.

    Args:
        text: Text block from a voter card
        pdf_file: PDF filename for error logging
        page_num: Page number for error logging
        error_logger: Error logger instance
        _retry: Internal flag to prevent infinite recursion

    Returns:
        List of extracted names (strings)
    """
    all_names = []

    # Split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Find all name lines (both voter names and relationship names)
    for line in lines:
        # Skip lines that are clearly not names
        if any(skip in line for skip in ['घर क्रमाक', 'वय :', 'लिंग :', 'छायाचित्र', 'उपलब्ध', 'यादी भाग', 'दिनांकास', 'विधानसभा', 'विभाग क्रमांक']):
            continue

        # Skip standalone numbers or IDs
        if re.match(r'^[\dA-Z\s\-]+$', line.strip()) and len(line.strip()) < 20:
            continue

        # Check if this line contains any names with "नाव" pattern
        if ('नाव' in line or 'नाब' in line) and (':' in line or ';' in line):
            # Try to extract names from this line
            # Handle multiple names on one line (side-by-side cards)

            # Split by "नाव" occurrences to separate multiple entries
            parts = re.split(r'(?=(?:पतीचे\s*)?(?:वडिलांचे\s*)?(?:पत्नीचे\s*)?(?:आईचे\s*)?(?:नाव|नाब)\s*[:;])', line)

            for part in parts:
                if 'नाव' in part or 'नाब' in part:
                    # Try different label patterns
                    name = None
                    for label in ['पतीचे', 'वडिलांचे', 'पत्नीचे', 'आईचे', 'नाव', 'नाब']:
                        if label in part:
                            extracted = extract_name_from_line(part, label)
                            if extracted:
                                name = extracted
                                break

                    if name:
                        # Clean up the name
                        name = re.sub(r'^\s*(नाव|नाब|पतीचे|वडिलांचे|पत्नीचे|आईचे)\s*[:;]\s*', '', name)
                        name = re.sub(r'\s+', ' ', name).strip()
                        # Remove OCR artifacts
                        name = re.sub(r'\s*(att|Fart|Fert)\s*', ' ', name).strip()
                        name = re.sub(r'\s+', ' ', name).strip()
                        # Remove prefixes like 'ः' or other punctuation
                        name = re.sub(r'^[:\s;]+', '', name).strip()

                        # Validate the name
                        is_valid, rejection_reason = is_valid_devanagari_name(name)

                        if is_valid:
                            all_names.append(name)
                        elif name and len(name) >= 3:  # Only log if it's substantial enough
                            error_logger.log_rejected_name(pdf_file, page_num, name, rejection_reason)

    # Log errors for problematic extractions
    if not all_names and text and 'नाव' in text and not _retry:
        # Only try correction once to avoid recursion
        corrected_text = apply_ocr_corrections(text)
        if corrected_text != text:
            return extract_voter_from_text_block(corrected_text, pdf_file, page_num, error_logger, _retry=True)
        else:
            # Only log if there was 'नाव' in the text but we couldn't extract
            error_logger.log_error(pdf_file, page_num, text, "Could not extract any names from text block")

    return all_names


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


def extract_voters_from_page(page, page_num: int, pdf_file: str, error_logger: ErrorLogger) -> List[str]:
    """
    Extract voter information from a PDF page using OCR and text parsing.

    Args:
        page: pdfplumber page object
        page_num: Page number
        pdf_file: PDF filename for error logging
        error_logger: Error logger instance

    Returns:
        List of extracted names
    """
    names = []

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
            block_names = extract_voter_from_text_block(block, pdf_file, page_num, error_logger)
            names.extend(block_names)

    return names


def extract_voters_from_pdf(pdf_path: str, start_page: int, end_page: int, error_logger: ErrorLogger) -> List[str]:
    """
    Extract voter names from PDF file using spatial parsing.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)
        error_logger: Error logger instance

    Returns:
        List of extracted names
    """
    all_names = []
    pdf_file = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        actual_end = min(end_page, total_pages)

        for page_num in range(start_page - 1, actual_end):
            print(f"    Page {page_num + 1}/{actual_end}...", end=" ")
            page = pdf.pages[page_num]

            names = extract_voters_from_page(page, page_num + 1, pdf_file, error_logger)
            all_names.extend(names)
            print(f"{len(names)} names")

    return all_names


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
    print(f"Maharashtra Voter Name Extractor - All Names")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Output format: Plain text (one name per line)")
    print(f"Validation: Strict Devanagari only")
    print("=" * 80)

    # Initialize error logger
    error_file = output_file.replace('.txt', '_errors.txt').replace('.csv', '_errors.txt')
    error_logger = ErrorLogger(error_file)

    all_names = []
    successful = 0
    failed = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        try:
            names = extract_voters_from_pdf(str(pdf_file), start_page, end_page, error_logger)

            if names:
                all_names.extend(names)
                print(f"  ✓ Extracted {len(names)} names")
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

    # Combine all results and remove duplicates
    if all_names:
        initial_count = len(all_names)

        # Remove duplicates while preserving order
        unique_names = list(dict.fromkeys(all_names))
        duplicates_removed = initial_count - len(unique_names)

        # Save to TXT file
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in unique_names:
                f.write(name + '\n')

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"✓ Total PDFs processed: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"✓ Total names extracted: {initial_count}")
        print(f"✓ Duplicates removed: {duplicates_removed}")
        print(f"✓ Unique names: {len(unique_names)}")
        print(f"✓ Output saved to: {output_file}")
        print("=" * 80)

        # Display sample
        print("\nSample of extracted names (first 15):")
        print("-" * 80)
        for name in unique_names[:15]:
            print(f"  {name}")
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
        help='Output TXT filename (default: extracted_names_<timestamp>.txt)'
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
        output_file = f"extracted_names_{timestamp}.txt"

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
