# run_tests_live.py — test suite requiring a live server
#
# Collects only from directories that run_tests.py excludes:
#   tests/manual/       — live integration tests (need real API keys / server)
#   tests/performance/  — stability benchmarks
#
# NOTE: tests/manual/ scripts are standalone (not pytest-discoverable).
# Run them directly:
#   python tests/manual/test_models_availability.py
#   python tests/manual/test_session_live.py
#   python tests/manual/test_context_live.py

import sys
import pytest

answer = input("Is the local RelayFreeLLM server running? (y/N): ").strip().lower()
if answer != "y":
    print("Aborted. Start the server first, then re-run.")
    sys.exit(0)

args = [
    "tests/manual",
    "tests/performance",
    "-v",
]

sys.exit(pytest.main(args))
