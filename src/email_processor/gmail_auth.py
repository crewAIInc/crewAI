"""Gmail authentication and configuration handler"""
import os
import json
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from typing import Optional

class GmailAuthManager:
    """Manages Gmail API authentication and credentials"""

    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    TOKEN_FILE = 'token.pickle'
    CREDENTIALS_FILE = 'credentials.json'

    @classmethod
    def get_credentials(cls) -> Optional[Credentials]:
        """Get valid credentials, requesting user authentication if necessary."""
        creds = None

        # Load existing token if available
        if os.path.exists(cls.TOKEN_FILE):
            with open(cls.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        # If credentials are invalid or don't exist, refresh or create new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(cls.CREDENTIALS_FILE):
                    raise Exception(
                        f"Missing {cls.CREDENTIALS_FILE}. Please provide OAuth2 credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    cls.CREDENTIALS_FILE, cls.SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save credentials for future use
            with open(cls.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        return creds

    @classmethod
    def clear_credentials(cls) -> None:
        """Clear stored credentials"""
        if os.path.exists(cls.TOKEN_FILE):
            os.remove(cls.TOKEN_FILE)
