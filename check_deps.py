#!/usr/bin/env python3

print("Checking optional dependencies availability:")

try:
    import chromadb
    print('chromadb: AVAILABLE')
except ImportError:
    print('chromadb: NOT AVAILABLE')

try:
    import pdfplumber
    print('pdfplumber: AVAILABLE')
except ImportError:
    print('pdfplumber: NOT AVAILABLE')

try:
    import pyvis
    print('pyvis: AVAILABLE')
except ImportError:
    print('pyvis: NOT AVAILABLE')

try:
    import opentelemetry
    print('opentelemetry: AVAILABLE')
except ImportError:
    print('opentelemetry: NOT AVAILABLE')

try:
    import auth0
    print('auth0: AVAILABLE')
except ImportError:
    print('auth0: NOT AVAILABLE')

try:
    import aisuite
    print('aisuite: AVAILABLE')
except ImportError:
    print('aisuite: NOT AVAILABLE')
