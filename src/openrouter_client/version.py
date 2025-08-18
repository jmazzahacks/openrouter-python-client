"""
Version information for OpenRouter Client.

This module provides version information for the package.

Exported:
- __version__: Package version string
"""

# Package version
__version__ = "0.0.4"

# Minimum supported API version
__api_version__ = "v1"

# Version details for debugging and compatibility checks
VERSION_INFO = {
    "package": __version__,
    "api": __api_version__,
    "python": ">=3.9",
    "smartsurge": ">=0.0.4",
}
