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
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import Levenshtein


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
        self.names_needing_review = []  # OCR-corrected names that need manual verification
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

    def log_name_needs_review(self, pdf_file: str, page_num: int, original_name: str,
                               corrected_name: str, reason: str):
        """Log a name that was partially corrected but needs manual review."""
        self.names_needing_review.append({
            'pdf_file': pdf_file,
            'page_number': page_num,
            'original_name': original_name,
            'corrected_name': corrected_name,
            'reason': reason
        })

    def save(self):
        """Save errors and rejected names to file."""
        if self.errors or self.rejected_names or self.names_needing_review:
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

                if self.names_needing_review:
                    f.write(f"\n\nNAMES NEEDING MANUAL REVIEW ({len(self.names_needing_review)} total)\n")
                    f.write("=" * 80 + "\n")
                    f.write("These names were partially corrected but require human verification\n\n")
                    for idx, review_item in enumerate(self.names_needing_review, 1):
                        f.write(f"\n--- Review Item #{idx} ---\n")
                        f.write(f"File: {review_item['pdf_file']}\n")
                        f.write(f"Page: {review_item['page_number']}\n")
                        f.write(f"Original: {review_item['original_name']}\n")
                        f.write(f"Corrected: {review_item['corrected_name']}\n")
                        f.write(f"Reason: {review_item['reason']}\n")
                        f.write("-" * 80 + "\n")

            total = len(self.errors) + len(self.rejected_names) + len(self.names_needing_review)
            print(f"  ⚠ Logged {total} issues to: {self.output_path}")
            if self.names_needing_review:
                print(f"  📝 {len(self.names_needing_review)} names flagged for manual review")


def extract_first_name(full_name: str) -> list:
    """
    Extract first name(s) from a full name based on word count rules.
    Returns a list of individual words to be written on separate lines.

    NOTE: OCR correction is now done in post-processing via phonetic clustering,
    not during extraction.

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

    # Extract first name(s) based on word count (no correction applied here)
    if word_count == 1:
        return words
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


def transliterate_to_latin(devanagari_text: str) -> str:
    """
    Transliterate Devanagari text to Latin script (ITRANS) for phonetic comparison.

    Args:
        devanagari_text: Text in Devanagari script

    Returns:
        Transliterated text in Latin script
    """
    try:
        return transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except:
        return devanagari_text


def phonetic_similarity(name1: str, name2: str) -> float:
    """
    Calculate phonetic similarity between two names using transliteration distance.

    Args:
        name1: First name in Devanagari
        name2: Second name in Devanagari

    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    # Transliterate both to Latin for phonetic comparison
    latin1 = transliterate_to_latin(name1).lower()
    latin2 = transliterate_to_latin(name2).lower()

    # Calculate Levenshtein distance
    distance = Levenshtein.distance(latin1, latin2)
    max_len = max(len(latin1), len(latin2))

    if max_len == 0:
        return 1.0

    # Convert distance to similarity score (0-1)
    similarity = 1.0 - (distance / max_len)
    return similarity


def build_phonetic_clusters(names_list: List[str], max_distance: int = 1) -> Dict[int, List[Tuple[str, str, int]]]:
    """
    Build clusters of phonetically similar names using transliteration + edit distance.

    Strategy:
    1. First group names by character length (only compare names of same/similar length)
    2. Within each length group, cluster phonetically with strict threshold (distance ≤ 1)

    Args:
        names_list: List of unique names in Devanagari
        max_distance: Maximum Levenshtein distance to consider names similar (default: 1)

    Returns:
        Dictionary mapping cluster_id to list of (name, latin_transliteration, count) tuples
        Only returns clusters with more than one unique variant
    """
    from collections import Counter, defaultdict

    # Count occurrences of each name
    name_counts = Counter(names_list)
    unique_names = list(name_counts.keys())

    # Transliterate all names to Latin for comparison
    name_to_latin = {}
    for name in unique_names:
        name_to_latin[name] = transliterate_to_latin(name).lower()

    # STEP 1: Group names by transliterated length
    names_by_length = defaultdict(list)
    for name in unique_names:
        latin = name_to_latin[name]
        # Group by length, but allow ±1 character difference to catch minor variations
        length_key = len(latin)
        names_by_length[length_key].append(name)

    # STEP 2: Within each length group, cluster phonetically
    # Ensure ALL names in a cluster are within max_distance of EACH OTHER (not just the first)
    clusters_list = []

    for length, names_in_group in names_by_length.items():
        if len(names_in_group) < 2:
            continue  # No clustering needed for single name

        processed = set()

        for i, name1 in enumerate(names_in_group):
            if name1 in processed:
                continue

            latin1 = name_to_latin[name1]
            cluster = [(name1, latin1, name_counts[name1])]
            cluster_latins = [latin1]  # Track all Latin transliterations in this cluster
            processed.add(name1)

            # Find all names similar to ALL names already in the cluster
            for name2 in names_in_group[i+1:]:
                if name2 in processed:
                    continue

                latin2 = name_to_latin[name2]

                # Only cluster if lengths are very close (within 1 character)
                len_diff = abs(len(latin1) - len(latin2))
                if len_diff > 1:
                    continue

                # Check if name2 is within max_distance of ALL names in current cluster
                is_close_to_all = True
                for cluster_latin in cluster_latins:
                    distance = Levenshtein.distance(cluster_latin, latin2)
                    if distance > max_distance:
                        is_close_to_all = False
                        break

                # Only add to cluster if close to ALL existing members
                if is_close_to_all and len(cluster_latins) > 0:
                    # Double-check at least one distance is > 0 (not exact match)
                    min_dist = min(Levenshtein.distance(cl, latin2) for cl in cluster_latins)
                    if min_dist > 0:
                        cluster.append((name2, latin2, name_counts[name2]))
                        cluster_latins.append(latin2)
                        processed.add(name2)

            # Only save clusters with more than one variant
            if len(cluster) > 1:
                # Sort by count (descending) so most common variant is first
                cluster.sort(key=lambda x: x[2], reverse=True)
                clusters_list.append(cluster)

    # Convert to dictionary with IDs
    variant_clusters = {i+1: cluster for i, cluster in enumerate(clusters_list)}
    return variant_clusters


def apply_correction_if_needed(name: str, correction_type: str) -> str:
    """
    Apply corrections to a name based on the specified correction type.

    Args:
        name: Original name in Devanagari
        correction_type: 'phonetic_dict', 'pattern', 'both', or None

    Returns:
        Corrected name (or original if no correction applies)
    """
    if correction_type is None:
        return name

    # Detect what corrections are available for this name
    pattern_flags, suggested = detect_ocr_pattern_errors(name)

    if not pattern_flags:
        return name  # No correction available

    # Apply based on correction type
    if correction_type == 'phonetic_dict':
        # Only apply if correction is from phonetic dictionary
        if 'phonetic_dict' in pattern_flags:
            return suggested
    elif correction_type == 'pattern':
        # Only apply if correction is pattern-based (not phonetic_dict)
        if any(flag != 'phonetic_dict' for flag in pattern_flags):
            return suggested
    elif correction_type == 'both':
        # Apply any correction
        return suggested

    return name  # No matching correction type


def validate_devanagari_morphology(name: str) -> Tuple[bool, List[str]]:
    """
    Validate if a Devanagari name follows typical morphological patterns.

    Args:
        name: Name in Devanagari script

    Returns:
        Tuple of (is_valid, list_of_error_reasons)
    """
    error_reasons = []

    if not name or len(name.strip()) == 0:
        return False, ["Empty name"]

    name = name.strip()

    # Check 1: Starts with dependent vowel marks (these should never be at the start)
    # Dependent vowels: ि ी ु ू ृ ॄ ॅ े ै ॉ ो ौ ं ः ् ँ
    dependent_vowels = ['ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ', 'ं', 'ः', '्', 'ँ']
    if any(name.startswith(dv) for dv in dependent_vowels):
        error_reasons.append(f"Starts with dependent vowel mark: '{name[0]}'")

    # Check 2: Very short names (likely fragments)
    if len(name) <= 2:
        error_reasons.append(f"Very short ({len(name)} chars), likely fragment")

    # Check 3: Ends with halant (्) without following consonant (incomplete word)
    if name.endswith('्'):
        error_reasons.append("Ends with halant (्), incomplete word")

    # Check 4: Orphaned consonant clusters (प्पा, क्का without context)
    # If name is just a consonant cluster with no other content
    if len(name) <= 4 and '्' in name:
        consonants_with_halant = name.count('्')
        if consonants_with_halant >= len(name) / 2:
            error_reasons.append("Appears to be orphaned consonant cluster")

    # Check 5: Unusual symbols in wrong positions
    # nukta (़) should only appear after specific consonants
    if '़' in name:
        # Check if nukta appears in isolation or at start
        if name.startswith('़') or ' ़' in name:
            error_reasons.append("Nukta (़) in invalid position")

    # Check 6: Multiple spaces or special punctuation that indicates fragment
    if '  ' in name or any(p in name for p in ['॰', '।', '॥']):
        error_reasons.append("Contains unusual spacing or punctuation")

    # Check 7: Contains only vowel marks (no consonants or independent vowels)
    # Remove all dependent vowels and see if anything is left
    temp = name
    for dv in dependent_vowels:
        temp = temp.replace(dv, '')
    if len(temp.strip()) == 0:
        error_reasons.append("Contains only vowel marks, no base characters")

    is_valid = len(error_reasons) == 0
    return is_valid, error_reasons


def detect_ocr_pattern_errors(name: str) -> Tuple[List[str], str]:
    """
    Detect common OCR error patterns in Devanagari names WITHOUT auto-correcting.
    This function only flags patterns that could be corrected, for post-clustering analysis.

    Args:
        name: Name string to analyze

    Returns:
        Tuple of (pattern_flags, suggested_correction)
        - pattern_flags: List of pattern names that matched (e.g., ['ण्या→प्पा', 'त्त→त'])
        - suggested_correction: What the corrected name would be (for display in error file)
    """
    if not name:
        return [], ""

    pattern_flags = []
    suggested = name

    # Pattern 1: ण्या/ण्पा/णप्पा → प्पा patterns (very common OCR error)
    if 'ण्या' in name:
        pattern_flags.append('ण्या→प्पा')
        suggested = suggested.replace('ण्या', 'प्पा')
    elif 'ण्पा' in name:
        pattern_flags.append('ण्पा→प्पा')
        suggested = suggested.replace('ण्पा', 'प्पा')
    elif 'णप्पा' in name and not name.startswith('ण'):
        pattern_flags.append('णप्पा→प्पा')
        suggested = suggested.replace('णप्पा', 'प्पा')

    # Pattern 2: ंवाण्या, ंधाण्या patterns
    if 'ाण्' in name:
        pattern_flags.append('ाण्→ाप्')
        suggested = suggested.replace('ाण्या', 'ाप्पा').replace('ाण्पा', 'ाप्पा')

    # Pattern 3: च → व at word end (common OCR confusion)
    if name.endswith('च') and len(name) > 2:
        if any(name.endswith(pattern) for pattern in ['राच', 'शीच', 'ाच', 'ीच']):
            pattern_flags.append('च→व')
            suggested = suggested[:-1] + 'व'

    # Pattern 4: Trailing anusvara/chandrabindu
    if name.endswith('ं'):
        pattern_flags.append('trailing_ं')
        suggested = suggested[:-1]

    # Pattern 5: Common phonetic corrections dictionary
    phonetic_corrections = {
        'हणमंत': 'हनुमंत',
        'हणमंता': 'हनुमंत',
        'हणमाण्णा': 'हनुमान्ना',
        'हृणमाण्णा': 'हनुमान्ना',
        'किंरण': 'किरण',
        'अजिंत': 'अजित',
        'रंबिराज': 'रणबीरराज',
        'रंबिंद्र': 'रवींद्र',
        'सोम्लिंग': 'सोमलिंग',
        'अनित्ता': 'अनिता',
        'अनित्रा': 'अनिता',
        'शशिक्तांत': 'शशिकांत',
        'सांगेता': 'संगीता',
        'माथुरी': 'माधुरी',
        'शितल': 'शीतल',
        'पोर्णिमा': 'पूर्णिमा',
        'भींमा': 'भीमा',
        'निल्लास': 'निलास',
        'मोहून': 'मोहन',
        'जवश्री': 'जयश्री',
        'जवकुमार': 'जयकुमार',
        'चूनुस': 'युनुस',
        'पुस्तफा': 'मुस्तफा',
        'जुयेर': 'जुबेर',
        'रॉफेक': 'राफिक',
        'शौकत्त': 'शौकत',
        'यशवंत्त': 'यशवंत',
        'जयवंत्त': 'जयवंत',
        'हणमंत्त': 'हनुमंत',
    }

    if name in phonetic_corrections:
        pattern_flags.append('phonetic_dict')
        suggested = phonetic_corrections[name]

    # Pattern 6: Suffix corrections
    if name.endswith('त्त'):
        pattern_flags.append('त्त→त')
        suggested = suggested[:-2] + 'त'
    elif name.endswith('राध'):
        pattern_flags.append('राध→राम')
        suggested = suggested[:-1] + 'म'
    elif name.endswith('नाध'):
        pattern_flags.append('नाध→नाथ')
        suggested = suggested[:-1] + 'थ'

    return pattern_flags, suggested


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
            # Extract first name(s) - now returns just a list (no correction at this stage)
            first_names = extract_first_name(name)

            # Push full name
            names_queue.put({
                'pdf_file': pdf_basename,
                'name': name,
                'type': 'full',
                'worker_id': worker_id
            })

            # Push first name(s) if they exist
            # Note: Validation and correction now happens in post-processing
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
                       dedup_window_size: int,
                       apply_corrections: str = None):
    """
    Dedicated process for writing names to files with deduplication.

    Creates a single global only_names.txt file with cross-file deduplication,
    while maintaining per-PDF extracted_names.txt files.

    Args:
        names_queue: Queue from which to read extracted names
        error_queue: Queue to monitor worker completion
        output_dir: Directory where output files will be written
        num_workers: Total number of file worker processes
        dedup_window_size: Size of the sliding window for duplicate detection
    """
    try:
        # Track file handles for per-PDF full names
        file_handles: Dict[str, Dict[str, any]] = {}

        # Global first names collection (IN MEMORY - don't write file yet)
        global_first_names_list = []  # Collect all names in memory
        global_first_names_window = deque(maxlen=dedup_window_size)
        global_first_names_tracker = DuplicateTracker()
        global_first_count = 0

        # Track which PDFs contributed each first name (for cross-file statistics)
        first_name_sources: Dict[str, Set[str]] = {}  # normalized_name -> set of pdf_files

        error_logs_to_consolidate: Dict[str, List[str]] = {}  # pdf_file -> list of error log paths
        completed_workers = 0
        total_names_written = 0

        print(f"[Name Writer] Started (window size: {dedup_window_size}, normalization: NFD)")
        print(f"[Name Writer] Collecting names in memory for validation (not writing yet)")

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

                    # Initialize file handles for this PDF if not exists (only full names per-PDF)
                    if pdf_file not in file_handles:
                        full_names_file = str(output_dir / f"{pdf_file}_extracted_names.txt")

                        file_handles[pdf_file] = {
                            'full_handle': open(full_names_file, 'w', encoding='utf-8'),
                            'full_window': deque(maxlen=dedup_window_size),
                            'full_count': 0,
                            'full_tracker': DuplicateTracker()
                        }
                        print(f"[Name Writer] Created full names output file for: {pdf_file}")

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
                        # Collect first names in memory (don't write to file yet)
                        normalized = normalize_devanagari(name)
                        if normalized not in global_first_names_window:
                            # Store in memory instead of writing to file
                            global_first_names_list.append(name)
                            global_first_names_window.append(normalized)
                            global_first_names_tracker.add_name(name)
                            global_first_count += 1

                            # Track which PDF this name came from
                            if normalized not in first_name_sources:
                                first_name_sources[normalized] = set()
                            first_name_sources[normalized].add(pdf_file)
                        else:
                            # Name already exists, track cross-file duplicate
                            if normalized not in first_name_sources:
                                first_name_sources[normalized] = set()
                            first_name_sources[normalized].add(pdf_file)

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
                normalized = normalize_devanagari(name)

                if name_type == 'full':
                    # Full names require per-PDF file handles
                    if pdf_file in file_handles:
                        handles = file_handles[pdf_file]
                        if normalized not in handles['full_window']:
                            handles['full_handle'].write(name + '\n')
                            handles['full_window'].append(normalized)
                            handles['full_tracker'].add_name(name)
                            handles['full_count'] += 1
                            total_names_written += 1
                            remaining_count += 1

                elif name_type == 'first':
                    # Collect first names in memory (don't write to file yet)
                    if normalized not in global_first_names_window:
                        global_first_names_list.append(name)
                        global_first_names_window.append(normalized)
                        global_first_names_tracker.add_name(name)
                        global_first_count += 1
                        remaining_count += 1

                        # Track source PDF
                        if normalized not in first_name_sources:
                            first_name_sources[normalized] = set()
                        first_name_sources[normalized].add(pdf_file)
                    else:
                        # Track cross-file duplicate even if not collecting
                        if normalized not in first_name_sources:
                            first_name_sources[normalized] = set()
                        first_name_sources[normalized].add(pdf_file)

            except:
                break

        if remaining_count > 0:
            print(f"[Name Writer] Processed {remaining_count} remaining names")

        # Clean up temporary error logs (if any)
        print("\n[Name Writer] Cleaning up temporary error logs...")
        for pdf_file, error_log_paths in error_logs_to_consolidate.items():
            if error_log_paths:
                for error_log_path in error_log_paths:
                    try:
                        Path(error_log_path).unlink()
                    except:
                        pass

        # Close all file handles, generate duplicate reports, and print summary
        print("\n" + "=" * 80)
        print("NAME WRITER SUMMARY")
        print("=" * 80)

        # Per-PDF full names summary
        for pdf_file, handles in file_handles.items():
            print(f"\n{pdf_file}:")
            print(f"  Full names: {handles['full_count']}")

            # Close file handles
            handles['full_handle'].close()

            # Generate duplicate report for full names
            full_dup_report = str(output_dir / f"{pdf_file}_extracted_names_duplicates_report.txt")
            handles['full_tracker'].save_report(full_dup_report)

        # Global first names summary (before validation)
        print(f"\n{'Global First Names (collected in memory)':}")
        print(f"  Total unique first names collected: {global_first_count}")

        # Calculate cross-file duplicate statistics
        cross_file_names = {name: sources for name, sources in first_name_sources.items() if len(sources) > 1}
        print(f"  Names appearing in multiple PDFs: {len(cross_file_names)}")

        # ========================================================================
        # VALIDATION: Validate ALL names before writing to file
        # ========================================================================
        print("\n" + "=" * 80)
        print("VALIDATION: Analyzing all names before writing to only_names.txt")
        print("=" * 80)

        # Use the collected names from memory
        all_names_original = global_first_names_list
        print(f"[Validation] Analyzing {len(all_names_original)} unique names...")

        # Step 0: Apply corrections if requested
        if apply_corrections:
            print(f"[Validation] Applying corrections (type: {apply_corrections})...")
            all_names_corrected = []
            corrections_applied = 0
            for name in all_names_original:
                corrected = apply_correction_if_needed(name, apply_corrections)
                all_names_corrected.append(corrected)
                if corrected != name:
                    corrections_applied += 1
            print(f"  ✓ Applied {corrections_applied} corrections")
            all_names = all_names_corrected
        else:
            all_names = all_names_original

        # Step 1: Build phonetic clusters (edit distance ≤ 1, grouped by length)
        print("[Validation] Building phonetic clusters (grouped by length, distance ≤ 1)...")
        phonetic_clusters = build_phonetic_clusters(all_names, max_distance=1)
        print(f"  ✓ Found {len(phonetic_clusters)} phonetic variant clusters")

        # Step 2: Validate Devanagari morphology for all names
        print("[Validation] Validating Devanagari morphology...")
        morphologically_invalid = []
        for name in all_names:
            is_valid, error_reasons = validate_devanagari_morphology(name)
            if not is_valid:
                morphologically_invalid.append((name, error_reasons))
        print(f"  ✓ Found {len(morphologically_invalid)} morphologically invalid names")

        # Step 3: Detect OCR pattern errors (only for reporting, corrections already applied)
        print("[Validation] Detecting OCR pattern errors...")
        pattern_errors = []
        # If corrections were applied, check original names for reporting
        names_to_check = all_names_original if apply_corrections else all_names
        for name in names_to_check:
            pattern_flags, suggested = detect_ocr_pattern_errors(name)
            if pattern_flags:
                pattern_errors.append((name, pattern_flags, suggested))
        print(f"  ✓ Found {len(pattern_errors)} names with OCR pattern errors")

        # Step 4: Collect ALL invalid names
        print("\n[Validation] Determining which names to exclude from only_names.txt...")
        invalid_names_set = set()

        # Add all names from phonetic clusters (all variants need review)
        for cluster_id, variants in phonetic_clusters.items():
            for name, latin, count in variants:
                invalid_names_set.add(name)

        # Add morphologically invalid names
        for name, reasons in morphologically_invalid:
            invalid_names_set.add(name)

        # Add pattern error names
        for name, flags, suggested in pattern_errors:
            invalid_names_set.add(name)

        # Step 5: Split names into valid vs invalid
        valid_names = [name for name in all_names if name not in invalid_names_set]
        invalid_count = len(all_names) - len(valid_names)

        print(f"  ✓ Total names analyzed: {len(all_names)}")
        print(f"  ✓ Valid names (will write to only_names.txt): {len(valid_names)}")
        print(f"  ✓ Invalid names (will write to error file): {invalid_count}")

        # Step 6: Write only_names.txt with VALID names only
        only_names_file = str(output_dir / "only_names.txt")
        print(f"\n[Validation] Writing only_names.txt with {len(valid_names)} valid names...")
        with open(only_names_file, 'w', encoding='utf-8') as f:
            for name in valid_names:
                f.write(name + '\n')
        print(f"  ✓ only_names.txt written successfully")

        # Generate consolidated error file with 3 sections
        only_names_errors_file = str(output_dir / "only_names_errors.txt")
        total_flagged = len(phonetic_clusters) + len(morphologically_invalid) + len(pattern_errors)

        print(f"\n[Validation] Generating error report with {invalid_count} invalid names...")
        with open(only_names_errors_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CONSOLIDATED NAMES NEEDING MANUAL REVIEW\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total items flagged: {total_flagged}\n")
            f.write(f"  - Phonetic variant clusters: {len(phonetic_clusters)}\n")
            f.write(f"  - Morphologically invalid: {len(morphologically_invalid)}\n")
            f.write(f"  - Pattern-based corrections: {len(pattern_errors)}\n")
            f.write("\n")

            # ---- SECTION 1: Phonetic Variants ----
            f.write("=" * 80 + "\n")
            f.write("SECTION 1: PHONETIC VARIANTS\n")
            f.write("=" * 80 + "\n")
            f.write("These names are phonetically similar (grouped by length, edit distance ≤ 1).\n")
            f.write("Review to determine which variant is correct.\n\n")

            if phonetic_clusters:
                for cluster_id, variants in sorted(phonetic_clusters.items()):
                    f.write(f"--- Cluster #{cluster_id} ({len(variants)} variants) ---\n")

                    # Show all variants with their counts and transliterations
                    for name, latin, count in variants:
                        f.write(f"  • {name} → [{latin}] (appears {count} times)\n")

                    # Calculate edit distances between variants
                    if len(variants) > 1:
                        f.write(f"  Edit distances:\n")
                        for i in range(len(variants) - 1):
                            name1, latin1, _ = variants[i]
                            name2, latin2, _ = variants[i + 1]
                            distance = Levenshtein.distance(latin1, latin2)
                            f.write(f"    {latin1} ↔ {latin2} = {distance}\n")

                    f.write("-" * 80 + "\n")
            else:
                f.write("  (No phonetic variants found)\n\n")

            # ---- SECTION 2: Morphologically Invalid ----
            f.write("\n" + "=" * 80 + "\n")
            f.write("SECTION 2: MORPHOLOGICALLY INVALID\n")
            f.write("=" * 80 + "\n")
            f.write("These names violate Devanagari morphology rules.\n")
            f.write("They may be OCR errors, fragments, or require correction.\n\n")

            if morphologically_invalid:
                for idx, (name, error_reasons) in enumerate(morphologically_invalid, 1):
                    f.write(f"--- Item #{idx} ---\n")
                    f.write(f"  Name: {name}\n")
                    f.write(f"  Issues:\n")
                    for reason in error_reasons:
                        f.write(f"    - {reason}\n")
                    f.write("-" * 80 + "\n")
            else:
                f.write("  (No morphologically invalid names found)\n\n")

            # ---- SECTION 3: Pattern-Based Corrections ----
            f.write("\n" + "=" * 80 + "\n")
            f.write("SECTION 3: PATTERN-BASED CORRECTIONS\n")
            f.write("=" * 80 + "\n")
            f.write("These names match known OCR error patterns.\n")
            f.write("Suggested corrections are provided (applied AFTER phonetic clustering).\n\n")

            if pattern_errors:
                for idx, (name, pattern_flags, suggested) in enumerate(pattern_errors, 1):
                    f.write(f"--- Item #{idx} ---\n")
                    f.write(f"  Original: {name}\n")
                    f.write(f"  Suggested: {suggested}\n")
                    f.write(f"  Patterns: {', '.join(pattern_flags)}\n")
                    f.write("-" * 80 + "\n")
            else:
                f.write("  (No pattern-based errors found)\n\n")

        print(f"  ✓ Error report generated: {only_names_errors_file}")
        print(f"  ✓ Total items flagged: {total_flagged}")

        # Generate global first names duplicate report with cross-file statistics
        global_first_dup_report = str(output_dir / "only_names_duplicates_report.txt")
        global_first_names_tracker.save_report(global_first_dup_report)

        # Generate cross-file statistics report
        if cross_file_names:
            cross_file_report = str(output_dir / "only_names_cross_file_report.txt")
            with open(cross_file_report, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("CROSS-FILE FIRST NAMES REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Names appearing in multiple PDF files: {len(cross_file_names)}\n\n")

                # Sort by number of PDFs (descending) then by name
                sorted_names = sorted(cross_file_names.items(),
                                    key=lambda x: (-len(x[1]), x[0]))

                for normalized_name, pdf_sources in sorted_names:
                    f.write(f"\n{normalized_name}\n")
                    f.write(f"  Found in {len(pdf_sources)} PDF(s):\n")
                    for pdf in sorted(pdf_sources):
                        f.write(f"    - {pdf}\n")

            print(f"\n  Cross-file report generated: {cross_file_report}")

            # Show top 5 most common cross-file names
            print(f"\n  Top names appearing in multiple PDFs:")
            for normalized_name, pdf_sources in sorted_names[:5]:
                print(f"    {normalized_name}: {len(pdf_sources)} PDFs")

        print(f"\nTotal full names written: {total_names_written}")
        print("=" * 80)

    except Exception as e:
        print(f"[Name Writer] Fatal error: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Ensure all file handles are closed
        for handles in file_handles.values():
            try:
                handles['full_handle'].close()
            except:
                pass


def process_folder_parallel(folder_path: Path, start_page: int, end_page: int,
                           num_workers: int = None, dedup_window_size: int = None,
                           output_folder: Path = None, apply_corrections: str = None):
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
        args=(names_queue, error_queue, output_folder, len(pdf_files), dedup_window_size, apply_corrections)
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
        description='Extract voter names from Maharashtra voters list PDFs (Devanagari script) using parallel processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -f ./pdfs
  %(prog)s -f /path/to/pdfs -o ./output

  # Custom options
  %(prog)s -f ./pdfs -o ./results --pages 3-32
  %(prog)s -f ./pdfs -o ./output --workers 4 --dedup-window 500
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
        help='Output folder path for all generated files (default: ./output). All output files including only_names.txt, per-PDF files, and error files will be written here.'
    )

    parser.add_argument(
        '-p', '--pages',
        type=str,
        default='3-32',
        help='Page range to extract (default: 3-32). Format: START-END'
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help=f'Number of parallel worker processes (default: CPU count - 1 = {config.NUM_WORKERS})'
    )

    parser.add_argument(
        '--dedup-window',
        type=int,
        default=None,
        help=f'Deduplication window size (default: {config.DEDUP_WINDOW_SIZE})'
    )

    parser.add_argument(
        '-a', '--add-correction',
        type=str,
        choices=['phonetic_dict', 'pattern', 'both'],
        default=None,
        help='Auto-apply corrections during extraction: phonetic_dict (dictionary-based), pattern (pattern-based), or both'
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

    # Get input folder path
    folder_path = Path(args.folder).resolve()
    if not folder_path.exists():
        print(f"✗ Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"✗ Error: Path is not a directory: {folder_path}")
        sys.exit(1)

    # Determine output folder path
    if args.output:
        output_folder_path = Path(args.output).resolve()
    else:
        # Default to ./output in current directory
        output_folder_path = Path.cwd() / 'output'

    # Set worker and deduplication parameters
    num_workers = args.workers if args.workers else config.NUM_WORKERS
    dedup_window = args.dedup_window if args.dedup_window else config.DEDUP_WINDOW_SIZE

    # Validate parameters
    if num_workers < 1:
        print("✗ Error: Number of workers must be at least 1")
        sys.exit(1)

    if dedup_window < 1:
        print("✗ Error: Deduplication window size must be at least 1")
        sys.exit(1)

    # Process folder using parallel mode
    process_folder_parallel(folder_path, start_page, end_page, num_workers, dedup_window, output_folder_path, args.add_correction)


if __name__ == "__main__":
    main()
