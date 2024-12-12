"""
Test script to verify email_processor package installation and imports.
"""
from email_processor import (
    EmailAnalysisCrew,
    ResponseCrew,
    GmailTool,
    EmailTool,
    __version__
)

def test_package_installation():
    print(f"Email Processor Package Version: {__version__}")
    print("Successfully imported all components:")
    print(f" - {EmailAnalysisCrew.__name__}")
    print(f" - {ResponseCrew.__name__}")
    print(f" - {GmailTool.__name__}")
    print(f" - {EmailTool.__name__}")

if __name__ == "__main__":
    test_package_installation()
