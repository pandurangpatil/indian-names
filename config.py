"""
Configuration settings for the parallel voter name extraction system.
"""

import multiprocessing
import os


# Parallel Processing Configuration
def get_default_workers():
    """Get default number of worker processes (CPU count - 1)."""
    cpu_count = multiprocessing.cpu_count()
    # Reserve one CPU for the system and name writer process
    return max(1, cpu_count - 1)


NUM_WORKERS = get_default_workers()

# Deduplication Configuration
DEDUP_WINDOW_SIZE = 1000  # Number of recent names to check for duplicates

# Queue Configuration
QUEUE_TIMEOUT = 1  # Timeout in seconds for queue operations
QUEUE_MAX_SIZE = 10000  # Maximum queue size before blocking

# Sentinel value to signal worker completion
WORKER_SENTINEL = None

# Progress Update Interval
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N names

# Output File Configuration
DEFAULT_OUTPUT_FOLDER = './output'  # Default output folder in current directory
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Error Handling
MAX_RETRIES = 3  # Maximum retries for failed PDF processing
RETRY_DELAY = 1  # Delay in seconds between retries
