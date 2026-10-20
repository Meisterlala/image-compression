#!/usr/bin/env python3
"""Compatibility entry point for the full v5 reference interoperability check.

The reduced encoder has been removed. This now verifies full reference golden
streams and actual Rust-trained outputs in both directions. Pass --reference
and optionally --write-fixtures as documented in verify-coolchic-streams.py.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("verify-coolchic-streams.py")), run_name="__main__")
