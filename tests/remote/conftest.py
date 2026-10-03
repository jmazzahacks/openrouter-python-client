"""
Load API keys for the remote integration tests from the repo-root .env file.

Runs before the test modules are imported, so their module-level
skipif checks on OPENROUTER_API_KEY see the loaded values. Variables
already set in the environment take precedence over .env.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture(scope="session", autouse=True)
def require_api_key():
    """Skip the remote suite before fixtures can construct unauthenticated clients."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")


@pytest.fixture(scope="session")
def provisioning_api_key():
    """Gate only tests that exercise account management endpoints."""
    if not os.getenv("OPENROUTER_PROVISIONING_API_KEY"):
        pytest.skip("OPENROUTER_PROVISIONING_API_KEY is not set")
