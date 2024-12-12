"""
CrewAI Email Processing System
=============================

A system for intelligent email processing and response generation using CrewAI.
"""

from .email_analysis_crew import EmailAnalysisCrew
from .response_crew import ResponseCrew
from .gmail_tool import GmailTool
from .email_tool import EmailTool

__version__ = "0.1.0"
__all__ = ['EmailAnalysisCrew', 'ResponseCrew', 'GmailTool', 'EmailTool']
