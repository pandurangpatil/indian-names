# Parallel Processing Feature

## Overview

The voter name extraction tool now supports parallel processing to significantly speed up the extraction of names from multiple PDF files. This feature leverages multiprocessing to utilize all available CPU cores efficiently.

## Key Features

### 1. File-Level Parallelization
- Multiple PDF files are processed concurrently by worker processes
- Each worker process handles one PDF file at a time
- Configurable number of worker processes (default: CPU count - 1)

### 2. Centralized Name Writer
- Single dedicated process manages all name writing
- Prevents file conflicts and ensures data integrity
- Real-time writing as names are extracted

### 3. Sliding Window Deduplication
- **Full names**: Per-PDF sliding window for duplicate detection
- **First names**: Single global sliding window with cross-file deduplication
- Configurable window size (default: 1000 names)
- More efficient than end-of-processing global deduplication
- Names are checked against the last N names already written

### 4. Output File Structure
- **Per-PDF files**:
  - `{pdf_basename}_extracted_names.txt` - Full names (unique per PDF)
- **Global files**:
  - `only_names.txt` - First names only (deduplicated across all PDFs)
- Easy traceability from input to output for full names
- Single consolidated first names file prevents cross-file duplicates

## Usage

### Parallel Mode (Default)

```bash
# Basic usage - uses all available cores minus one
python extract_voter_names.py -f /path/to/pdfs

# Specify custom number of workers
python extract_voter_names.py -f /path/to/pdfs --workers 4

# Customize deduplication window
python extract_voter_names.py -f /path/to/pdfs --dedup-window 500

# Combine options
python extract_voter_names.py -f /path/to/pdfs --workers 8 --dedup-window 2000 --pages 3-32
```

### Sequential Mode (Original Behavior)

```bash
# Disable parallel processing
python extract_voter_names.py -f /path/to/pdfs --no-parallel --output names.txt

# Sequential mode with page range
python extract_voter_names.py -f /path/to/pdfs --no-parallel --output names.txt --pages 3-32
```

## Configuration

### Default Settings (config.py)

```python
NUM_WORKERS = cpu_count() - 1  # Auto-detected
DEDUP_WINDOW_SIZE = 1000        # Names
QUEUE_MAX_SIZE = 10000          # Messages
QUEUE_TIMEOUT = 1               # Seconds
```

### CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--parallel` | - | Enable parallel processing | `True` |
| `--no-parallel` | - | Disable parallel processing | `False` |
| `--workers` | `-w` | Number of worker processes | `CPU count - 1` |
| `--dedup-window` | - | Deduplication window size | `1000` |

## Architecture

### Process Structure

```
Main Process
├── Name Writer Process (single)
│   ├── Manages per-PDF full name file handles
│   ├── Manages global first names file (only_names.txt)
│   ├── Maintains per-PDF deduplication windows for full names
│   ├── Maintains global deduplication window for first names
│   └── Writes names in real-time
│
└── File Worker Processes (configurable)
    ├── Worker 1 → PDF File 1
    ├── Worker 2 → PDF File 2
    ├── Worker 3 → PDF File 3
    └── ...
```

### Communication Flow

```
File Workers → Names Queue → Name Writer → Output Files
File Workers → Error Queue → Main Process → Console
```

### Deduplication Logic

The name writer maintains separate deduplication strategies:

**For Full Names (Per-PDF)**:
- Each PDF maintains its own window of the last 1000 full names
- Duplicates are only checked within the same PDF
- Written to `{pdf_basename}_extracted_names.txt`

**For First Names (Global)**:
- Single global window of the last 1000 first names across all PDFs
- Duplicates are checked across all source PDFs
- Written to single `only_names.txt` file
- Tracks which PDFs contributed each name for cross-file statistics

When a new name arrives:
1. Check if it exists in the corresponding window (per-PDF for full, global for first)
2. If not found, write to file and add to window
3. If found, skip (it's a duplicate within the window)

The window automatically maintains only the last N names using a `deque` with `maxlen`.

## Performance

### Expected Speedup

Assuming:
- OCR processing takes 10-15 seconds per page (dominant bottleneck)
- 4-8 CPU cores available
- Multiple PDF files to process

Expected speedup:
- **2 workers**: ~1.8-2x faster
- **4 workers**: ~3.5-4x faster
- **8 workers**: ~6-8x faster

### Limitations

- Speedup is limited by the number of PDF files
  - If you have only 2 PDFs, using 8 workers won't help
- OCR is CPU-intensive, so I/O is rarely the bottleneck
- Memory usage increases with more workers

## Output Files

### Parallel Mode

For each input PDF `filename.pdf`, creates:
```
filename_extracted_names.txt              # Full names (per-PDF)
filename_extracted_names_duplicates_report.txt  # Duplicate report for full names
```

Additionally, creates global files:
```
only_names.txt                            # First names (deduplicated across all PDFs)
only_names_duplicates_report.txt          # Duplicate report for first names
only_names_cross_file_report.txt          # Report showing names appearing in multiple PDFs
```

### Sequential Mode

Creates two files with custom or timestamped names:
```
extracted_names_TIMESTAMP.txt   # Full names (default)
only_names_TIMESTAMP.txt        # First names only
```

Or with `--output names.txt`:
```
names.txt                       # Full names
only_names.txt                  # First names only
```

## Error Handling

### Worker Errors

- Each worker logs errors to temporary files in `/tmp/`
- Format: `/tmp/{pdf_basename}_errors_worker_{worker_id}.txt`
- Errors include:
  - Extraction failures
  - Rejected names with reasons
  - OCR issues

### Process Management

- Graceful shutdown on Ctrl+C
- Automatic cleanup of file handles
- Timeout protection for name writer (30 seconds)
- Workers continue even if one fails

## Examples

### Example 1: Process All PDFs with Maximum Speed

```bash
python extract_voter_names.py -f ./pdfs --pages 3-32
```

Output:
```
# Per-PDF files
2024-EROLLGEN-S13-282-FinalRoll-Revision1-MAR-1-WI_extracted_names.txt
2024-EROLLGEN-S13-287-FinalRoll-Revision1-MAR-190-WI_extracted_names.txt
...

# Global files
only_names.txt                       # All first names, deduplicated across PDFs
only_names_duplicates_report.txt     # Duplicate statistics
only_names_cross_file_report.txt     # Names appearing in multiple PDFs
```

### Example 2: Limited Workers for Background Processing

```bash
python extract_voter_names.py -f ./pdfs --workers 2 --pages 3-32
```

Uses only 2 workers, leaving more CPU for other tasks.

### Example 3: Conservative Deduplication

```bash
python extract_voter_names.py -f ./pdfs --dedup-window 100
```

Only checks last 100 names for duplicates (faster, but may miss some duplicates).

### Example 4: Aggressive Deduplication

```bash
python extract_voter_names.py -f ./pdfs --dedup-window 5000
```

Checks last 5000 names for duplicates (slower, but catches more duplicates).

### Example 5: Sequential Mode for Comparison

```bash
python extract_voter_names.py -f ./pdfs --no-parallel --output combined_names.txt
```

Uses the original sequential processing and combines all names into a single file.

## Troubleshooting

### Issue: "Too many open files" error

**Cause**: Each PDF requires 1 file handle (full names), plus 1 global handle (first names)

**Solution**:
- Reduce number of workers: `--workers 2`
- Increase system limit: `ulimit -n 4096`

### Issue: High memory usage

**Cause**: Large deduplication windows and many concurrent workers

**Solution**:
- Reduce window size: `--dedup-window 500`
- Reduce workers: `--workers 4`

### Issue: Name writer timeout

**Cause**: Queue has too many pending names

**Solution**:
- Increase timeout in `config.py`
- Reduce workers to slow down production

### Issue: Missing names in output

**Check**:
1. Worker error logs in `/tmp/`
2. Console output for rejection reasons
3. Error log files created alongside output files

## Comparison: Parallel vs Sequential

| Feature | Parallel Mode | Sequential Mode |
|---------|--------------|-----------------|
| **Speed** | 4-8x faster | Baseline |
| **Output** | Per-PDF full names + Global first names | Single combined |
| **Deduplication** | Per-PDF for full, Global for first | Global, at end |
| **Cross-file Dedup** | Yes (first names) | Yes |
| **Memory** | Higher | Lower |
| **Traceability** | Excellent | Good |
| **Resumability** | Can reprocess failed files | Must restart all |
| **Default** | Yes | No |

## Best Practices

1. **Use parallel mode by default** - It's faster and more organized
2. **Adjust workers based on CPU cores** - Don't exceed CPU count
3. **Use large dedup windows for exhaustive extraction** - 1000+ is good
4. **Use small dedup windows for speed** - 100-500 for quick processing
5. **Monitor worker error logs** - Check `/tmp/` for issues
6. **Keep page ranges reasonable** - 30-50 pages per PDF is optimal
7. **Use sequential mode only when you need a single combined file**

## Technical Details

### Multiprocessing Safety

- Uses `multiprocessing.Queue` for inter-process communication
- Each process has isolated memory space
- File handles are not shared between processes
- Thread-safe queue operations

### Deduplication Implementation

```python
from collections import deque

# Create a fixed-size window
window = deque(maxlen=1000)

# Check and add
if name not in window:
    file.write(name + '\n')
    window.append(name)  # Automatically evicts oldest if full
```

### Process Coordination

1. Main process spawns N file workers + 1 name writer
2. File workers push names to shared queue
3. File workers push completion status to error queue
4. Name writer monitors both queues
5. Name writer exits when all workers report completion
6. Main process waits for all processes to join

## Future Enhancements

Potential improvements:
- Page-level parallelization within a single PDF
- Progress bars with `tqdm`
- Checkpoint/resume functionality
- Distributed processing across machines
- Real-time statistics dashboard
- Memory-mapped file writing for even faster I/O
