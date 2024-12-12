"""
Gmail integration tool for email processing flow.
Handles email fetching and thread context retrieval.
"""
from typing import Dict, List, Optional
from datetime import datetime
import base64
import email
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from crewai.tools import BaseTool
from .gmail_auth import GmailAuthManager

class GmailTool(BaseTool):
    """Tool for interacting with Gmail API"""
    name: str = "Gmail Tool"
    description: str = "Tool for interacting with Gmail API to fetch and process emails"

    def __init__(self):
        """Initialize Gmail API client"""
        super().__init__()
        self.credentials = GmailAuthManager.get_credentials()
        self.service = build('gmail', 'v1', credentials=self.credentials)

    def get_latest_emails(self, limit: int = 5) -> List[Dict]:
        """
        Get latest emails with thread context.

        Args:
            limit: Maximum number of emails to fetch (default: 5)

        Returns:
            List[Dict]: List of email data with context
        """
        try:
            # Get latest messages
            messages = self.service.users().messages().list(
                userId='me',
                maxResults=limit,
                labelIds=['INBOX'],
                q='is:unread'  # Focus on unread messages
            ).execute().get('messages', [])

            # Get full email data with context
            return [self._get_email_with_context(msg['id']) for msg in messages]
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []

    def _get_email_with_context(self, message_id: str) -> Dict:
        """
        Get full email data with thread context.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict: Email data with thread context
        """
        try:
            # Get full message data
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            # Get thread data
            thread_id = message.get('threadId')
            thread = self.service.users().threads().get(
                userId='me',
                id=thread_id
            ).execute()

            # Extract headers
            headers = {
                header['name'].lower(): header['value']
                for header in message['payload']['headers']
            }

            # Parse message content
            content = self._get_message_content(message['payload'])

            # Extract sender information
            sender = self._parse_email_address(headers.get('from', ''))

            return {
                'id': message_id,
                'thread_id': thread_id,
                'subject': headers.get('subject', ''),
                'sender': sender,
                'to': self._parse_email_address(headers.get('to', '')),
                'date': headers.get('date'),
                'content': content,
                'labels': message.get('labelIds', []),
                'thread_messages': [
                    self._parse_thread_message(msg)
                    for msg in thread.get('messages', [])
                    if msg['id'] != message_id  # Exclude current message
                ],
                'thread_size': len(thread.get('messages', [])),
                'is_unread': 'UNREAD' in message.get('labelIds', [])
            }
        except Exception as e:
            print(f"Error getting email context: {e}")
            return {}

    def _get_message_content(self, payload: Dict) -> str:
        """Extract message content from payload"""
        if 'body' in payload and payload['body'].get('data'):
            return base64.urlsafe_b64decode(
                payload['body']['data'].encode('ASCII')
            ).decode('utf-8')

        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    if 'data' in part['body']:
                        return base64.urlsafe_b64decode(
                            part['body']['data'].encode('ASCII')
                        ).decode('utf-8')
        return ''

    def _parse_thread_message(self, message: Dict) -> Dict:
        """Parse thread message into simplified format"""
        headers = {
            header['name'].lower(): header['value']
            for header in message['payload']['headers']
        }

        return {
            'id': message['id'],
            'sender': self._parse_email_address(headers.get('from', '')),
            'date': headers.get('date'),
            'content': self._get_message_content(message['payload']),
            'labels': message.get('labelIds', [])
        }

    def _parse_email_address(self, address: str) -> Dict:
        """Parse email address string into components"""
        if '<' in address and '>' in address:
            name = address[:address.find('<')].strip()
            email_addr = address[address.find('<')+1:address.find('>')]
            return {'name': name, 'email': email_addr}
        return {'name': '', 'email': address.strip()}

    def mark_as_read(self, message_id: str) -> bool:
        """Mark email as read"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            print(f"Error marking message as read: {e}")
            return False

    def send_response(self,
                     to: str,
                     subject: str,
                     message_text: str,
                     thread_id: Optional[str] = None) -> bool:
        """
        Send email response.

        Args:
            to: Recipient email address
            subject: Email subject
            message_text: Response content
            thread_id: Optional thread ID for reply

        Returns:
            bool: True if sent successfully
        """
        try:
            message = MIMEText(message_text)
            message['to'] = to
            message['subject'] = subject

            # Create message
            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')

            body = {'raw': raw_message}
            if thread_id:
                body['threadId'] = thread_id

            self.service.users().messages().send(
                userId='me',
                body=body
            ).execute()
            return True
        except Exception as e:
            print(f"Error sending response: {e}")
            return False

    def _run(self, method: str = "get_latest_emails", **kwargs) -> Dict:
        """Required implementation of BaseTool._run"""
        try:
            if method == "get_latest_emails":
                return self.get_latest_emails(kwargs.get("limit", 5))
            elif method == "get_thread_history":
                return self.get_thread_history(kwargs.get("thread_id"))
            elif method == "get_sender_info":
                return self.get_sender_info(kwargs.get("email"))
            elif method == "mark_as_read":
                return self.mark_as_read(kwargs.get("message_id"))
            elif method == "send_response":
                return self.send_response(
                    to=kwargs.get("to"),
                    subject=kwargs.get("subject"),
                    message_text=kwargs.get("message_text"),
                    thread_id=kwargs.get("thread_id")
                )
            return None
        except Exception as e:
            print(f"Error in GmailTool._run: {e}")
            return None
