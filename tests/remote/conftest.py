"""
Load API keys for the remote integration tests from the repo-root .env file.

Runs before the test modules are imported, so their module-level
skipif checks on OPENROUTER_API_KEY see the loaded values. Variables
already set in the environment take precedence over .env.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
