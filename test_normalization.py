#!/usr/bin/env python3
"""Test script to verify normalization on the test file."""

import unicodedata
from collections import defaultdict

def normalize_devanagari(name: str) -> str:
    """Normalize Devanagari names for duplicate detection."""
    if not name:
        return ''

    # Apply NFD normalization first
    normalized = unicodedata.normalize('NFD', name)

    # Remove common diacritical marks
    normalized = normalized.replace('\u0902', '')  # Anusvara ं
    normalized = normalized.replace('\u0901', '')  # Chandrabindu ँ
    normalized = normalized.replace('\u093C', '')  # Nukta ़

    # Normalize vowel lengths
    vowel_mappings = {
        '\u093F': '\u0940',  # ि -> ी
        '\u0941': '\u0942',  # ु -> ू
        '\u0947': '\u0948',  # े -> ै
        '\u094B': '\u094C',  # ो -> ौ
    }

    for short, long in vowel_mappings.items():
        normalized = normalized.replace(short, long)

    # Apply NFC to recompose
    normalized = unicodedata.normalize('NFC', normalized)

    return normalized

# Read test file
test_file = 'test/2024-EROLLGEN-S13-282-FinalRoll-Revision1-MAR-1-WI_only_names.txt'

print('=' * 80)
print('DUPLICATE DETECTION TEST')
print('=' * 80)
print(f'Input file: {test_file}')
print()

# Track normalized names
duplicates = defaultdict(list)

with open(test_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        # Skip empty lines
        name = line.strip()
        if not name:
            continue

        # Normalize and track
        normalized = normalize_devanagari(name)
        duplicates[normalized].append((line_num, name))

# Analyze results
total_names = sum(len(occurrences) for occurrences in duplicates.values())
unique_names = len(duplicates)
duplicate_groups = [(norm, occ) for norm, occ in duplicates.items() if len(occ) > 1]
total_duplicates = sum(len(occ) - 1 for _, occ in duplicate_groups)

print('SUMMARY:')
print('-' * 80)
print(f'Total names: {total_names}')
print(f'Unique names (after normalization): {unique_names}')
print(f'Duplicate groups found: {len(duplicate_groups)}')
print(f'Total duplicates removed: {total_duplicates}')
print(f'Duplicate rate: {total_duplicates/total_names*100:.2f}%')
print()

# Show first 15 duplicate groups
print('SAMPLE DUPLICATE GROUPS (first 15):')
print('=' * 80)

for idx, (normalized, occurrences) in enumerate(sorted(duplicate_groups, key=lambda x: len(x[1]), reverse=True)[:15], 1):
    print(f'\n{idx}. Normalized: {normalized}')
    print(f'   Count: {len(occurrences)} occurrences')
    for line_num, original in occurrences:
        print(f'   Line {line_num:3d}: {original}')
