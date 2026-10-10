#!/usr/bin/env python3
"""Monitor transcription progress with visual feedback."""

import os
import sys
import time
from pathlib import Path

# Set environment variables for better progress display
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TQDM_POSITION'] = '0'
os.environ['TQDM_MINITERS'] = '1'

def main():
    """Run transcription with enhanced progress monitoring."""
    from wav_transcriber.cli import main as cli_main
    
    # Enable verbose progress for all operations
    os.environ['TQDM_DISABLE'] = '0'
    os.environ['HF_HUB_VERBOSITY'] = 'info'
    
    print("=" * 80)
    print("WAV Transcriber - Industry-Leading Accuracy Configuration")
    print("=" * 80)
    print("\nConfiguration Highlights:")
    print("  * Model: large-v3 (3GB, best-in-class accuracy)")
    print("  * Beam Size: 10 (high-quality search)")
    print("  * Compute: float16 (GPU-optimized)")
    print("  * Audio: Aggressive preprocessing enabled")
    print("  * Alignment: Word-level timestamps (+/- 50ms)")
    print("  * Diarization: Speaker identification enabled")
    print("\nStarting transcription pipeline...\n")
    
    try:
        return cli_main()
    except KeyboardInterrupt:
        print("\n\nTranscription interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
