"""
Main workflow script for email processing system.
Implements event-based automation for email analysis and response generation.
"""
from typing import Dict
import logging
from datetime import datetime

from .email_flow import EmailProcessingFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_emails() -> Dict:
    """
    Main workflow function.
    Implements event-based email processing using CrewAI Flow.

    Returns:
        Dict: Processing results including counts and any errors
    """
    try:
        logger.info("Starting email processing workflow")

        # Initialize and start flow
        flow = EmailProcessingFlow()
        results = flow.kickoff()

        # Log processing results
        logger.info(
            f"Processed {results['processed_emails']} emails, "
            f"generated {results['responses_generated']} responses"
        )

        if results.get('errors', 0) > 0:
            logger.warning(f"Encountered {results['errors']} errors during processing")

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            **results
        }

    except Exception as e:
        logger.error(f"Email processing failed: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    results = process_emails()
    print(f"\nProcessing Results:\n{'-' * 20}")
    for key, value in results.items():
        print(f"{key}: {value}")
