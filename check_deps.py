#!/usr/bin/env python3

print("Checking optional dependencies availability:")

try:
    import chromadb  # noqa: F401
    print('chromadb: AVAILABLE')
except ImportError:
    print('chromadb: NOT AVAILABLE')

try:
    import pdfplumber  # noqa: F401
    print('pdfplumber: AVAILABLE')
except ImportError:
    print('pdfplumber: NOT AVAILABLE')

try:
    import pyvis  # noqa: F401
    print('pyvis: AVAILABLE')
except ImportError:
    print('pyvis: NOT AVAILABLE')

try:
    import opentelemetry  # noqa: F401
    print('opentelemetry: AVAILABLE')
except ImportError:
    print('opentelemetry: NOT AVAILABLE')

try:
    import auth0  # noqa: F401
    print('auth0: AVAILABLE')
except ImportError:
    print('auth0: NOT AVAILABLE')

try:
    import aisuite  # noqa: F401
    print('aisuite: AVAILABLE')
except ImportError:
    print('aisuite: NOT AVAILABLE')
