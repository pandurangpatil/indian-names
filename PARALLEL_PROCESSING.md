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
- Each output file maintains its own sliding window for duplicate detection
- Configurable window size (default: 1000 names)
- More efficient than end-of-processing global deduplication
- Names are checked against the last N names already written to the file

### 4. Per-File Output Mapping
- Each input PDF generates separate output files:
  - `{pdf_basename}_extracted_names.txt` - Full names
  - `{pdf_basename}_only_names.txt` - First names only
- Easy traceability from input to output
- Better organization for large batches

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
│   ├── Manages all file handles
│   ├── Maintains per-file deduplication windows
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

For each PDF file, the name writer maintains:
- **Full Names Window**: Last 1000 full names written for this PDF
- **First Names Window**: Last 1000 first names written for this PDF

When a new name arrives:
1. Check if it exists in the corresponding window
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
filename_extracted_names.txt    # Full names
filename_only_names.txt         # First names only
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
2024-EROLLGEN-S13-282-FinalRoll-Revision1-MAR-1-WI_extracted_names.txt
2024-EROLLGEN-S13-282-FinalRoll-Revision1-MAR-1-WI_only_names.txt
2024-EROLLGEN-S13-287-FinalRoll-Revision1-MAR-190-WI_extracted_names.txt
2024-EROLLGEN-S13-287-FinalRoll-Revision1-MAR-190-WI_only_names.txt
...
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

**Cause**: Each PDF requires 2 file handles (full names + first names)

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
| **Output** | Per-file | Single combined |
| **Deduplication** | Per-file, sliding window | Global, at end |
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
