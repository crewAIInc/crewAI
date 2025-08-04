from abc import ABC, abstractmethod
from crewai.cli.authentication.main import Oauth2Settings

class BaseProvider(ABC):
    def __init__(self, settings: Oauth2Settings):
        self.settings = settings

    @abstractmethod
    def get_authorize_url(self) -> str:
        ...

    @abstractmethod
    def get_token_url(self) -> str:
        ...

    @abstractmethod
    def get_jwks_url(self) -> str:
        ...

    @abstractmethod
    def get_issuer(self) -> str:
        ...

    @abstractmethod
    def get_audience(self) -> str:
        ...

    @abstractmethod
    def get_client_id(self) -> str:
        ...
