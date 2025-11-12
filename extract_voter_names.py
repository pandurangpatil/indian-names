#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract first name, middle name, and surname from Maharashtra voters list PDFs in Devanagari script.
Supports batch processing of multiple PDF files.
"""

import re
import argparse
import pdfplumber
import pandas as pd
from pathlib import Path
from typing import List, Dict
import pytesseract
import sys
from datetime import datetime


def is_valid_name(name_text: str) -> bool:
    """Check if the extracted text is a valid name and not a label."""
    exclude_patterns = [
        r'वडील', r'पत्नी', r'पती', r'आडनाव', r'विधानसभा', r'मतदारसंघ',
        r'क्रमांक', r'सांगली', r'महाराष्ट्र', r'छायाचित्र', r'उपलब्ध', r'प्रत',
    ]

    for pattern in exclude_patterns:
        if re.search(pattern, name_text):
            return False

    return len(name_text) >= 2


def extract_name_components(name_line: str) -> Dict[str, str]:
    """
    Extract first name, middle name, and surname from name line.

    Args:
        name_line: Line containing name in format "नाव : FirstName MiddleName Surname"

    Returns:
        Dictionary with first_name, middle_name, surname
    """
    # Remove "नाव :" prefix and labels
    name_text = re.sub(r'नाव\s*[:：]\s*', '', name_line).strip()
    name_text = re.sub(r'(पत्नी|पतीचे|वडील|आडनाव)\s*नाव\s*[:：]\s*', '', name_text).strip()
    name_text = re.sub(r'(पत्तीचे|वडिलांचे|यडिलांचे|चडिलांचे|झजाचाई|शणाप्रत्ताप)', '', name_text).strip()
    name_text = re.sub(r'^[-\s]+', '', name_text)

    # Split by spaces
    name_parts = [part.strip() for part in name_text.split() if part.strip()]

    if len(name_parts) == 0:
        return {"first_name": "", "middle_name": "", "surname": ""}
    elif len(name_parts) == 1:
        return {"first_name": name_parts[0], "middle_name": "", "surname": ""}
    elif len(name_parts) == 2:
        return {"first_name": name_parts[0], "middle_name": "", "surname": name_parts[1]}
    else:
        return {
            "first_name": name_parts[0],
            "middle_name": " ".join(name_parts[1:-1]),
            "surname": name_parts[-1]
        }


def extract_voters_from_page_ocr(page) -> List[Dict[str, str]]:
    """Extract voter information from a PDF page using OCR."""
    voters = []

    # Convert page to image
    img = page.to_image(resolution=300)
    pil_img = img.original

    # Perform OCR with Marathi language support
    try:
        text = pytesseract.image_to_string(pil_img, lang='mar+eng')
    except Exception as e:
        print(f"    OCR Error: {e}")
        return voters

    if not text:
        return voters

    # Process lines to find names
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()

        if 'नाव' in line and ':' in line:
            name_parts = extract_name_components(line)
            full_name = f"{name_parts['first_name']} {name_parts['middle_name']} {name_parts['surname']}".strip()

            if name_parts['first_name'] and is_valid_name(full_name):
                voters.append({
                    'first_name': name_parts['first_name'],
                    'middle_name': name_parts['middle_name'],
                    'surname': name_parts['surname']
                })

    return voters


def extract_voters_from_pdf(pdf_path: str, start_page: int, end_page: int) -> pd.DataFrame:
    """
    Extract voter names from PDF file using OCR.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)

    Returns:
        pandas DataFrame with extracted voter information
    """
    all_voters = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        actual_end = min(end_page, total_pages)

        for page_num in range(start_page - 1, actual_end):
            print(f"    Page {page_num + 1}/{actual_end}...", end=" ")
            page = pdf.pages[page_num]

            # Try text extraction first
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                # Text extraction works
                lines = text.split('\n')
                page_voters = []
                for line in lines:
                    if 'नाव' in line and ':' in line:
                        name_parts = extract_name_components(line)
                        if name_parts['first_name']:
                            page_voters.append({
                                'first_name': name_parts['first_name'],
                                'middle_name': name_parts['middle_name'],
                                'surname': name_parts['surname']
                            })
                voters = page_voters
            else:
                # Use OCR
                voters = extract_voters_from_page_ocr(page)

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
    print(f"Maharashtra Voter Name Extractor")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Using Marathi OCR (Tesseract)")
    print("=" * 80)

    all_names = []
    successful = 0
    failed = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        try:
            df = extract_voters_from_pdf(str(pdf_file), start_page, end_page)

            if not df.empty:
                all_names.append(df)
                print(f"  ✓ Extracted {len(df)} names")
                successful += 1
            else:
                print(f"  ⚠ No names extracted")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            continue

    # Combine all results
    if all_names:
        final_df = pd.concat(all_names, ignore_index=True)

        # Remove duplicates based on all three columns
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates()
        duplicates_removed = initial_count - len(final_df)

        # Save to CSV
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
        pd.set_option('display.width', 80)
        pd.set_option('display.max_colwidth', 25)
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
