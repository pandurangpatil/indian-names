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
from typing import List, Tuple, Dict, Optional
import pytesseract
import sys
from datetime import datetime
import multiprocessing
from collections import deque
import traceback
import time
import config
import unicodedata


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


def normalize_devanagari(name: str) -> str:
    """
    Normalize Devanagari names for consistent duplicate detection.

    Applies aggressive normalization to handle common variations:
    - Unicode NFD normalization
    - Remove anusvara (ं U+0902) and chandrabindu (ँ U+0901)
    - Remove nukta (़ U+093C) for Urdu/Persian sounds
    - Normalize vowel matras (short vs long: ि/ी, ु/ू, etc.)

    Args:
        name: Name string to normalize

    Returns:
        Normalized name string for comparison
    """
    if not name:
        return ""

    # Apply NFD normalization first
    normalized = unicodedata.normalize('NFD', name)

    # Remove common diacritical marks that cause variations
    # Anusvara (ं U+0902) - nasal sound modifier
    normalized = normalized.replace('\u0902', '')
    # Chandrabindu (ँ U+0901) - nasalization
    normalized = normalized.replace('\u0901', '')
    # Nukta (़ U+093C) - used for Persian/Urdu sounds
    normalized = normalized.replace('\u093C', '')

    # Normalize vowel length differences (treat short and long vowels as same)
    vowel_mappings = {
        '\u093F': '\u0940',  # ि (short i) -> ी (long i)
        '\u0941': '\u0942',  # ु (short u) -> ू (long u)
        '\u0947': '\u0948',  # े (short e) -> ै (ai)
        '\u094B': '\u094C',  # ो (short o) -> ौ (au)
    }

    for short, long in vowel_mappings.items():
        normalized = normalized.replace(short, long)

    # Apply NFC to recompose any decomposed characters
    normalized = unicodedata.normalize('NFC', normalized)

    return normalized


class DuplicateTracker:
    """Track duplicate names found during processing for reporting."""

    def __init__(self):
        # Maps normalized name -> list of (original_name, line_number) tuples
        self.duplicates: Dict[str, List[Tuple[str, int]]] = {}
        self.line_counter = 0

    def add_name(self, name: str) -> bool:
        """
        Track a name and check if it's a duplicate.

        Args:
            name: Original name to track

        Returns:
            True if this is the first occurrence (not a duplicate), False if duplicate
        """
        self.line_counter += 1
        normalized = normalize_devanagari(name)

        if normalized not in self.duplicates:
            # First occurrence
            self.duplicates[normalized] = [(name, self.line_counter)]
            return True
        else:
            # Duplicate found
            self.duplicates[normalized].append((name, self.line_counter))
            return False

    def get_duplicate_groups(self) -> List[Dict]:
        """
        Get all duplicate groups found.

        Returns:
            List of dictionaries containing duplicate information
        """
        duplicate_groups = []

        for normalized, occurrences in self.duplicates.items():
            if len(occurrences) > 1:
                duplicate_groups.append({
                    'normalized': normalized,
                    'count': len(occurrences),
                    'occurrences': occurrences
                })

        return duplicate_groups

    def get_statistics(self) -> Dict:
        """Get summary statistics about duplicates."""
        duplicate_groups = self.get_duplicate_groups()
        total_duplicates = sum(group['count'] - 1 for group in duplicate_groups)

        return {
            'total_names_processed': self.line_counter,
            'unique_names': len(self.duplicates),
            'duplicate_groups': len(duplicate_groups),
            'total_duplicates': total_duplicates
        }

    def save_report(self, output_path: str):
        """Save duplicate report to file."""
        duplicate_groups = self.get_duplicate_groups()

        if not duplicate_groups:
            return

        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DUPLICATE NAMES REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total names processed: {stats['total_names_processed']}\n")
            f.write(f"Unique names (after deduplication): {stats['unique_names']}\n")
            f.write(f"Duplicate groups found: {stats['duplicate_groups']}\n")
            f.write(f"Total duplicates removed: {stats['total_duplicates']}\n")
            f.write(f"Duplicate rate: {stats['total_duplicates']/stats['total_names_processed']*100:.2f}%\n")
            f.write("\n")

            f.write(f"DUPLICATE GROUPS ({len(duplicate_groups)} groups)\n")
            f.write("=" * 80 + "\n\n")

            # Sort by count (most duplicates first)
            duplicate_groups.sort(key=lambda x: x['count'], reverse=True)

            for idx, group in enumerate(duplicate_groups, 1):
                f.write(f"--- Group #{idx} ---\n")
                f.write(f"Normalized form: {group['normalized']}\n")
                f.write(f"Total occurrences: {group['count']}\n")
                f.write(f"Variations found:\n")

                for original, line_num in group['occurrences']:
                    f.write(f"  Line {line_num}: {original}\n")

                f.write("\n")

        print(f"  ✓ Duplicate report saved to: {output_path}")
        print(f"  ✓ Found {stats['duplicate_groups']} duplicate groups, removed {stats['total_duplicates']} duplicates")


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


def extract_first_name(full_name: str) -> list:
    """
    Extract first name(s) from a full name based on word count rules.
    Returns a list of individual words to be written on separate lines.

    Rules:
    - 1 word: return [word] (1 line)
    - 2 words: return [first_word] (1 line)
    - 3 words: return [first_word, second_word] (2 lines)
    - 4+ words: return [first_word] (1 line)

    Args:
        full_name: The full name string

    Returns:
        List of extracted name words (each word will be on a separate line)
    """
    words = full_name.strip().split()
    word_count = len(words)

    if word_count == 0:
        return []
    elif word_count == 1:
        return [words[0]]
    elif word_count == 2:
        return [words[0]]
    elif word_count == 3:
        return [words[0], words[1]]
    else:  # 4 or more words
        return [words[0]]


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


def clean_name(name: str) -> str:
    """
    Clean a name by removing label words, special characters, and other artifacts.

    Args:
        name: Raw name string potentially containing labels and special characters

    Returns:
        Cleaned name string
    """
    if not name:
        return ""

    # Remove prefix labels (at start of name)
    name = re.sub(r'^नावः\s*', '', name)
    name = re.sub(r"^नाव[',]?\s*", '', name)

    # Remove suffix labels (at end of name)
    # Order matters - check longer patterns first
    name = re.sub(r'\s+वडिलांचें$', '', name)
    name = re.sub(r'\s+वडिलांचे$', '', name)
    name = re.sub(r'\s+वडिलां$', '', name)
    name = re.sub(r'\s+डेलांचे$', '', name)
    name = re.sub(r'\s+पती$', '', name)
    name = re.sub(r'\s+नाव$', '', name)
    name = re.sub(r'\s+आई$', '', name)

    # Handle "उर्फ" (alias) patterns
    # If उर्फ appears in the middle, keep the part before it (primary name)
    if ' उर्फ ' in name:
        name = name.split(' उर्फ ')[0]
    # If उर्फ is at the start, remove it
    name = re.sub(r'^उर्फ\s+', '', name)

    # Remove possessive "च्या" constructions
    # Pattern: <name>च्या <surname> → keep only <surname>
    name = re.sub(r'[^\s]+च्या\s+', '', name)

    # Remove special characters
    name = re.sub(r'[ः]', '', name)  # Visarga
    name = re.sub(r"[',]", '', name)  # Apostrophe and comma

    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def is_valid_devanagari_name(name: str) -> Tuple[bool, str]:
    """
    Strictly validate if text is a valid Devanagari name.
    Cleans the name first, then validates.

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    # Clean the name first
    name = clean_name(name)
    name = name.strip()

    # Check minimum length after cleaning
    if len(name) < 3:
        return False, "Too short after cleaning (less than 3 characters)"

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
    # Allow: space, hyphen (-), nukta (़) which is legitimate in Urdu-origin names
    allowed_pattern = r'^[\u0900-\u097F\s\-]+$'
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
    punctuation_count = len(re.findall(r'[\-;:|]', name))
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
                        # Basic cleanup before validation
                        name = re.sub(r'^\s*(नाव|नाब|पतीचे|वडिलांचे|पत्नीचे|आईचे)\s*[:;]\s*', '', name)
                        name = re.sub(r'\s+', ' ', name).strip()
                        # Remove OCR artifacts
                        name = re.sub(r'\s*(att|Fart|Fert)\s*', ' ', name).strip()
                        name = re.sub(r'\s+', ' ', name).strip()
                        # Remove prefixes like ':' or other punctuation
                        name = re.sub(r'^[:\s;]+', '', name).strip()

                        # Validate the name (validation will clean it further)
                        is_valid, rejection_reason = is_valid_devanagari_name(name)

                        if is_valid:
                            # Get the cleaned version
                            cleaned_name = clean_name(name)
                            if cleaned_name:  # Only add if not empty after cleaning
                                all_names.append(cleaned_name)
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


def extract_voters_from_pdf(pdf_path: str, start_page: int, end_page: int, error_logger: ErrorLogger, worker_id: int = None) -> List[str]:
    """
    Extract voter names from PDF file using spatial parsing.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)
        error_logger: Error logger instance
        worker_id: Optional worker ID for parallel processing progress messages

    Returns:
        List of extracted names
    """
    all_names = []
    pdf_file = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        actual_end = min(end_page, total_pages)

        for page_num in range(start_page - 1, actual_end):
            worker_prefix = f"[Worker {worker_id}] " if worker_id is not None else ""
            print(f"{worker_prefix}    Page {page_num + 1}/{actual_end}...", end=" ")
            page = pdf.pages[page_num]

            names = extract_voters_from_page(page, page_num + 1, pdf_file, error_logger)
            all_names.extend(names)
            print(f"{len(names)} names")

    return all_names


def process_folder(folder_path: Path, start_page: int, end_page: int, output_file: str, output_folder: Path = None) -> None:
    """
    Process all PDF files in a folder and extract voter names.

    Args:
        folder_path: Path to folder containing PDF files
        start_page: Starting page number for extraction
        end_page: Ending page number for extraction
        output_file: Output CSV filename
        output_folder: Optional output folder path (default: None = use folder_path)
    """
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"✗ No PDF files found in: {folder_path}")
        return

    # Determine output location
    if output_folder is None:
        output_folder = folder_path
    else:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Update output file path to be in output folder
    output_file = str(output_folder / Path(output_file).name)

    print("=" * 80)
    print(f"Maharashtra Voter Name Extractor - All Names")
    print("=" * 80)
    print(f"Folder: {folder_path}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Output format: Plain text (one name per line)")
    print(f"Validation: Strict Devanagari only")
    print(f"Deduplication: Unicode NFD normalization")
    print("=" * 80)

    # Initialize error logger and duplicate tracker
    error_file = output_file.replace('.txt', '_errors.txt').replace('.csv', '_errors.txt')
    error_logger = ErrorLogger(error_file)
    duplicate_tracker = DuplicateTracker()

    all_names = []
    successful = 0
    failed = 0

    for idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
        try:
            names = extract_voters_from_pdf(str(pdf_file), start_page, end_page, error_logger, None)

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

    # Combine all results and remove duplicates using normalization
    if all_names:
        initial_count = len(all_names)

        # Remove duplicates using Unicode normalization
        unique_names = []
        for name in all_names:
            if duplicate_tracker.add_name(name):
                unique_names.append(name)

        duplicates_removed = initial_count - len(unique_names)

        # Save to TXT file
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in unique_names:
                f.write(name + '\n')

        # Extract and save only first names (flatten list of lists)
        only_names_file = output_file.replace('extracted_names_', 'only_names_')
        extracted_first_names = []
        for name in unique_names:
            name_words = extract_first_name(name)
            extracted_first_names.extend(name_words)

        with open(only_names_file, 'w', encoding='utf-8') as f:
            for name in extracted_first_names:
                if name:  # Only write non-empty names
                    f.write(name + '\n')

        # Save duplicate report
        duplicates_file = output_file.replace('.txt', '_duplicates_report.txt')
        duplicate_tracker.save_report(duplicates_file)

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"✓ Total PDFs processed: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"✓ Total names extracted: {initial_count}")
        print(f"✓ Duplicates removed: {duplicates_removed}")
        print(f"✓ Unique names: {len(unique_names)}")
        print(f"✓ Output saved to: {output_file}")
        print(f"✓ First names saved to: {only_names_file} ({len(extracted_first_names)} words)")
        print(f"✓ Duplicates report: {duplicates_file}")
        print("=" * 80)

        # Display sample
        print("\nSample of extracted names (first 15):")
        print("-" * 80)
        for name in unique_names[:15]:
            print(f"  {name}")
        print("-" * 80)

    else:
        print("\n✗ No names were extracted from any PDF files.")


def file_worker_process(pdf_path: str, start_page: int, end_page: int,
                       names_queue: multiprocessing.Queue,
                       error_queue: multiprocessing.Queue,
                       worker_id: int):
    """
    Worker process that extracts names from a single PDF file.

    Args:
        pdf_path: Path to the PDF file to process
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed)
        names_queue: Queue to push extracted names
        error_queue: Queue to push error messages
        worker_id: Unique identifier for this worker
    """
    try:
        pdf_file = Path(pdf_path)
        pdf_basename = pdf_file.stem

        # Create a temporary error logger for this worker
        temp_error_log = f"/tmp/{pdf_basename}_errors_worker_{worker_id}.txt"
        error_logger = ErrorLogger(temp_error_log)

        print(f"[Worker {worker_id}] Processing: {pdf_file.name}")

        # Extract names from the PDF
        names = extract_voters_from_pdf(str(pdf_path), start_page, end_page, error_logger, worker_id)

        # Push each name to the queue with metadata
        for name in names:
            # Extract first name(s) - returns a list
            first_names = extract_first_name(name)

            # Push full name
            names_queue.put({
                'pdf_file': pdf_basename,
                'name': name,
                'type': 'full',
                'worker_id': worker_id
            })

            # Push first name(s) if they exist
            if first_names:
                for first_name in first_names:
                    names_queue.put({
                        'pdf_file': pdf_basename,
                        'name': first_name,
                        'type': 'first',
                        'worker_id': worker_id
                    })

        # Save worker-specific error log
        error_logger.save()

        # Send completion message
        error_queue.put({
            'pdf_file': pdf_basename,
            'worker_id': worker_id,
            'status': 'completed',
            'names_count': len(names),
            'error_log': temp_error_log
        })

        print(f"[Worker {worker_id}] Completed: {pdf_file.name} ({len(names)} names)")

    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {str(e)}\n{traceback.format_exc()}"
        error_queue.put({
            'pdf_file': Path(pdf_path).stem,
            'worker_id': worker_id,
            'status': 'failed',
            'error': error_msg
        })
        print(f"[Worker {worker_id}] Failed: {Path(pdf_path).name} - {str(e)}")


def name_writer_process(names_queue: multiprocessing.Queue,
                       error_queue: multiprocessing.Queue,
                       output_dir: Path,
                       num_workers: int,
                       dedup_window_size: int):
    """
    Dedicated process for writing names to files with deduplication.

    Args:
        names_queue: Queue from which to read extracted names
        error_queue: Queue to monitor worker completion
        output_dir: Directory where output files will be written
        num_workers: Total number of file worker processes
        dedup_window_size: Size of the sliding window for duplicate detection
    """
    try:
        # Track file handles, deduplication windows, and duplicate trackers for each PDF
        file_handles: Dict[str, Dict[str, any]] = {}
        error_logs_to_consolidate: Dict[str, List[str]] = {}  # pdf_file -> list of error log paths
        completed_workers = 0
        total_names_written = 0

        print(f"[Name Writer] Started (window size: {dedup_window_size}, normalization: NFD)")

        while completed_workers < num_workers:
            try:
                # Check for completion messages
                try:
                    error_msg = error_queue.get_nowait()
                    if error_msg.get('status') in ['completed', 'failed']:
                        completed_workers += 1
                        print(f"[Name Writer] Worker {error_msg['worker_id']} {error_msg['status']} "
                              f"({completed_workers}/{num_workers})")

                        # Track error log for consolidation
                        if error_msg.get('status') == 'completed' and error_msg.get('error_log'):
                            pdf_file = error_msg['pdf_file']
                            if pdf_file not in error_logs_to_consolidate:
                                error_logs_to_consolidate[pdf_file] = []
                            error_logs_to_consolidate[pdf_file].append(error_msg['error_log'])
                except:
                    pass

                # Process names from queue
                try:
                    name_data = names_queue.get(timeout=config.QUEUE_TIMEOUT)

                    pdf_file = name_data['pdf_file']
                    name = name_data['name']
                    name_type = name_data['type']

                    # Initialize file handles and dedup window for this PDF if not exists
                    if pdf_file not in file_handles:
                        full_names_file = str(output_dir / f"{pdf_file}_extracted_names.txt")
                        first_names_file = str(output_dir / f"{pdf_file}_only_names.txt")

                        file_handles[pdf_file] = {
                            'full_handle': open(full_names_file, 'w', encoding='utf-8'),
                            'first_handle': open(first_names_file, 'w', encoding='utf-8'),
                            'full_window': deque(maxlen=dedup_window_size),
                            'first_window': deque(maxlen=dedup_window_size),
                            'full_count': 0,
                            'first_count': 0,
                            'full_tracker': DuplicateTracker(),
                            'first_tracker': DuplicateTracker()
                        }
                        print(f"[Name Writer] Created output files for: {pdf_file}")

                    handles = file_handles[pdf_file]

                    # Check for duplicates using normalization
                    if name_type == 'full':
                        normalized = normalize_devanagari(name)
                        if normalized not in handles['full_window']:
                            handles['full_handle'].write(name + '\n')
                            handles['full_handle'].flush()  # Ensure immediate write
                            handles['full_window'].append(normalized)
                            handles['full_tracker'].add_name(name)
                            handles['full_count'] += 1
                            total_names_written += 1

                    elif name_type == 'first':
                        normalized = normalize_devanagari(name)
                        if normalized not in handles['first_window']:
                            handles['first_handle'].write(name + '\n')
                            handles['first_handle'].flush()  # Ensure immediate write
                            handles['first_window'].append(normalized)
                            handles['first_tracker'].add_name(name)
                            handles['first_count'] += 1

                except multiprocessing.queues.Empty:
                    # Queue is empty, continue checking for completion
                    continue

            except KeyboardInterrupt:
                print("[Name Writer] Interrupted by user")
                break

        # Process any remaining names in the queue
        print("[Name Writer] Processing remaining names...")
        remaining_count = 0
        while True:
            try:
                name_data = names_queue.get_nowait()
                pdf_file = name_data['pdf_file']
                name = name_data['name']
                name_type = name_data['type']

                if pdf_file in file_handles:
                    handles = file_handles[pdf_file]
                    normalized = normalize_devanagari(name)

                    if name_type == 'full' and normalized not in handles['full_window']:
                        handles['full_handle'].write(name + '\n')
                        handles['full_window'].append(normalized)
                        handles['full_tracker'].add_name(name)
                        handles['full_count'] += 1
                        total_names_written += 1
                        remaining_count += 1

                    elif name_type == 'first' and normalized not in handles['first_window']:
                        handles['first_handle'].write(name + '\n')
                        handles['first_window'].append(normalized)
                        handles['first_tracker'].add_name(name)
                        handles['first_count'] += 1
                        remaining_count += 1

            except:
                break

        if remaining_count > 0:
            print(f"[Name Writer] Processed {remaining_count} remaining names")

        # Consolidate error logs and generate reports
        print("\n[Name Writer] Consolidating error logs and generating reports...")
        for pdf_file, error_log_paths in error_logs_to_consolidate.items():
            if error_log_paths:
                consolidated_error_file = str(output_dir / f"{pdf_file}_errors.txt")

                # Read and consolidate all error logs
                with open(consolidated_error_file, 'w', encoding='utf-8') as out_f:
                    for error_log_path in error_log_paths:
                        try:
                            with open(error_log_path, 'r', encoding='utf-8') as in_f:
                                out_f.write(in_f.read())
                                out_f.write("\n" + "=" * 80 + "\n")
                        except Exception as e:
                            print(f"[Name Writer] Warning: Could not read error log {error_log_path}: {e}")

                print(f"  ✓ Consolidated error log: {consolidated_error_file}")

                # Clean up temporary error logs
                for error_log_path in error_log_paths:
                    try:
                        Path(error_log_path).unlink()
                    except:
                        pass

        # Close all file handles, generate duplicate reports, and print summary
        print("\n" + "=" * 80)
        print("NAME WRITER SUMMARY")
        print("=" * 80)
        for pdf_file, handles in file_handles.items():
            print(f"\n{pdf_file}:")
            print(f"  Full names: {handles['full_count']}")
            print(f"  First names: {handles['first_count']}")

            # Close file handles
            handles['full_handle'].close()
            handles['first_handle'].close()

            # Generate duplicate reports
            full_dup_report = str(output_dir / f"{pdf_file}_extracted_names_duplicates_report.txt")
            first_dup_report = str(output_dir / f"{pdf_file}_only_names_duplicates_report.txt")

            handles['full_tracker'].save_report(full_dup_report)
            handles['first_tracker'].save_report(first_dup_report)

        print(f"\nTotal names written: {total_names_written}")
        print("=" * 80)

    except Exception as e:
        print(f"[Name Writer] Fatal error: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Ensure all handles are closed
        for handles in file_handles.values():
            try:
                handles['full_handle'].close()
                handles['first_handle'].close()
            except:
                pass


def process_folder_parallel(folder_path: Path, start_page: int, end_page: int,
                           num_workers: int = None, dedup_window_size: int = None,
                           output_folder: Path = None):
    """
    Process all PDF files in a folder using parallel workers.

    Args:
        folder_path: Path to folder containing PDF files
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed)
        num_workers: Number of worker processes (default: CPU count - 1)
        dedup_window_size: Size of deduplication window (default: from config)
        output_folder: Optional output folder path (default: None = use folder_path)
    """
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    if dedup_window_size is None:
        dedup_window_size = config.DEDUP_WINDOW_SIZE

    # Determine output location
    if output_folder is None:
        output_folder = folder_path
    else:
        output_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PARALLEL PROCESSING MODE")
    print("=" * 80)
    print(f"Worker processes: {num_workers}")
    print(f"Deduplication window: {dedup_window_size} names")
    print(f"Deduplication method: Unicode NFD normalization")
    print(f"Input folder: {folder_path}")
    print(f"Output folder: {output_folder}")
    print("=" * 80 + "\n")

    # Find all PDF files
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"✗ No PDF files found in: {folder_path}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process\n")

    # Create multiprocessing queues
    names_queue = multiprocessing.Queue(maxsize=config.QUEUE_MAX_SIZE)
    error_queue = multiprocessing.Queue()

    # Start the name writer process (use output_folder, not folder_path)
    writer_process = multiprocessing.Process(
        target=name_writer_process,
        args=(names_queue, error_queue, output_folder, len(pdf_files), dedup_window_size)
    )
    writer_process.start()

    # Create and start worker processes with proper concurrent execution
    pdf_queue = list(enumerate(pdf_files, 1))  # Queue of (worker_id, pdf_file) tuples
    active_workers = []

    print(f"[Main] Starting up to {num_workers} concurrent workers...\n")

    # Start initial batch of workers
    while pdf_queue and len(active_workers) < num_workers:
        worker_id, pdf_file = pdf_queue.pop(0)
        worker = multiprocessing.Process(
            target=file_worker_process,
            args=(str(pdf_file), start_page, end_page, names_queue, error_queue, worker_id)
        )
        worker.start()
        active_workers.append(worker)
        print(f"[Main] Started worker {worker_id} for {pdf_file.name}")

    # As workers finish, start new ones
    while pdf_queue or active_workers:
        # Remove finished workers
        for worker in active_workers[:]:  # Copy list to avoid modification issues
            if not worker.is_alive():
                worker.join()
                active_workers.remove(worker)

        # Start new workers if we have capacity and pending files
        while pdf_queue and len(active_workers) < num_workers:
            worker_id, pdf_file = pdf_queue.pop(0)
            worker = multiprocessing.Process(
                target=file_worker_process,
                args=(str(pdf_file), start_page, end_page, names_queue, error_queue, worker_id)
            )
            worker.start()
            active_workers.append(worker)
            print(f"[Main] Started worker {worker_id} for {pdf_file.name}")

        # Small sleep to prevent busy-waiting
        if active_workers:
            time.sleep(0.1)

    # All workers have completed
    print("\n[Main] All workers completed. Waiting for name writer to finish...")

    # Wait for name writer to complete
    writer_process.join(timeout=30)
    if writer_process.is_alive():
        print("[Main] Name writer taking too long, terminating...")
        writer_process.terminate()
        writer_process.join()

    print("\n✓ Parallel processing completed!")
    print(f"\nOutput files created in: {output_folder}")
    print("File naming format:")
    print("  <pdf_basename>_extracted_names.txt")
    print("  <pdf_basename>_only_names.txt")
    print("  <pdf_basename>_errors.txt")
    print("  <pdf_basename>_extracted_names_duplicates_report.txt")
    print("  <pdf_basename>_only_names_duplicates_report.txt")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract voter names from Maharashtra voters list PDFs (Devanagari script)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parallel processing (default, per-file output)
  %(prog)s --folder /path/to/pdfs
  %(prog)s -f ./pdfs --pages 3-32
  %(prog)s -f ./pdfs --workers 4 --dedup-window 500

  # Sequential processing (original behavior, single output file)
  %(prog)s -f ./pdfs --no-parallel --output names.txt
  %(prog)s -f . --no-parallel --output names.txt --pages 3-32
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
        help='Output TXT filename (default: extracted_names_<timestamp>.txt). Creates both extracted_names and only_names files. Only used in sequential mode.'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Output folder for all generated files (default: ./output in current directory). If not specified, files are saved in the input folder.'
    )

    parser.add_argument(
        '-p', '--pages',
        type=str,
        default='3-32',
        help='Page range to extract (default: 3-32). Format: START-END'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Use parallel processing (default: enabled)'
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        default=False,
        help='Disable parallel processing (use sequential mode)'
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: CPU count - 1 = {config.NUM_WORKERS})'
    )

    parser.add_argument(
        '--dedup-window',
        type=int,
        default=None,
        help=f'Deduplication window size (default: {config.DEDUP_WINDOW_SIZE})'
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

    # Determine output folder
    if args.output_folder:
        output_folder_path = Path(args.output_folder).resolve()
    else:
        # Default to ./output in current directory
        output_folder_path = Path.cwd() / 'output'

    # Determine processing mode
    use_parallel = args.parallel and not args.no_parallel

    # Process folder
    if use_parallel:
        # Use parallel processing
        num_workers = args.workers if args.workers else config.NUM_WORKERS
        dedup_window = args.dedup_window if args.dedup_window else config.DEDUP_WINDOW_SIZE

        # Validate workers count
        if num_workers < 1:
            print("✗ Error: Number of workers must be at least 1")
            sys.exit(1)

        # Validate dedup window
        if dedup_window < 1:
            print("✗ Error: Deduplication window size must be at least 1")
            sys.exit(1)

        process_folder_parallel(folder_path, start_page, end_page, num_workers, dedup_window, output_folder_path)
    else:
        # Use sequential processing (original behavior)
        print("\n" + "=" * 80)
        print("SEQUENTIAL PROCESSING MODE")
        print("=" * 80 + "\n")

        process_folder(folder_path, start_page, end_page, output_file, output_folder_path)


if __name__ == "__main__":
    main()
