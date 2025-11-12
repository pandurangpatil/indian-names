# Maharashtra Voter Name Extractor

Extract first name, middle name, and surname from Maharashtra voters list PDFs in Devanagari (Marathi) script.

## Features

- ✅ Uses Marathi OCR for accurate extraction
- ✅ Batch processing of multiple PDF files
- ✅ Simple 3-column CSV output (first_name, middle_name, surname)
- ✅ Automatic duplicate removal
- ✅ Configurable page range extraction
- ✅ Progress tracking for multiple files

## Requirements

- Python 3.8+
- Tesseract OCR with Marathi language support

## Installation

### 1. Install Tesseract

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-mar

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install pdfplumber pandas pytesseract Pillow
```

## Usage

### Basic Usage

Extract names from all PDFs in a folder:

```bash
python extract_voter_names.py --folder /path/to/pdfs
```

### Specify Output File

```bash
python extract_voter_names.py --folder /path/to/pdfs --output my_names.csv
```

### Custom Page Range

Extract only pages 3 to 10:

```bash
python extract_voter_names.py --folder /path/to/pdfs --pages 3-10
```

### Current Directory

Process PDFs in the current directory:

```bash
python extract_voter_names.py --folder .
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--folder` | `-f` | Path to folder containing PDF files | **(required)** |
| `--output` | `-o` | Output CSV filename | `extracted_names_<timestamp>.csv` |
| `--pages` | `-p` | Page range to extract (format: START-END) | `3-32` |
| `--help` | `-h` | Show help message | - |

## Output Format

The script generates a CSV file with 3 columns:

| first_name | middle_name | surname |
|------------|-------------|---------|
| प्रकाश | चंद्रकांत | चोपडे |
| मनीषा | सुरेखा | पिसाळ |
| राहुल | अप्पासाहेब | खंडागळे |

The CSV file is UTF-8 encoded with BOM for proper display in Excel and other spreadsheet applications.

## Examples

### Example 1: Single Folder

```bash
python extract_voter_names.py -f ./voter_pdfs -o all_voters.csv
```

### Example 2: Sample Pages

Test extraction on first few pages:

```bash
python extract_voter_names.py -f . -p 3-5 -o sample.csv
```

### Example 3: Full Extraction

Extract all pages from all PDFs:

```bash
python extract_voter_names.py -f /path/to/pdfs -p 3-32 -o full_list.csv
```

## How It Works

1. **PDF Detection**: Scans the specified folder for all `.pdf` files
2. **OCR Processing**: Uses Tesseract with Marathi language model to extract text
3. **Name Parsing**: Identifies lines containing "नाव :" and extracts name components
4. **Filtering**: Removes labels like "वडील", "पत्नी" and invalid entries
5. **Deduplication**: Removes duplicate names across all PDFs
6. **CSV Export**: Saves results to UTF-8 encoded CSV file

## Performance

- **OCR Speed**: ~10-15 seconds per page at 300 DPI
- **Accuracy**: Depends on PDF quality and scan resolution
- **Memory**: Processes pages sequentially to minimize memory usage

## Troubleshooting

### "Tesseract not found" error

Make sure Tesseract is installed and in your PATH:

```bash
which tesseract  # macOS/Linux
where tesseract  # Windows
```

### No names extracted

- Check if PDFs contain voter lists with "नाव :" pattern
- Verify pages contain Devanagari text
- Try adjusting page range with `--pages` option

### Poor OCR accuracy

- Ensure PDFs are high quality (not low-resolution scans)
- Verify Marathi language data is installed: `tesseract --list-langs | grep mar`

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
# indian-names
