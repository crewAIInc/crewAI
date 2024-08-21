from auth0.authentication.token_verifier import (
    AsymmetricSignatureVerifier,
    TokenVerifier,
)

from .constants import AUTH0_CLIENT_ID, AUTH0_DOMAIN


def validate_token(id_token: str) -> None:
    """
    Verify the token and its precedence

    :param id_token:
    """
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    issuer = f"https://{AUTH0_DOMAIN}/"
    signature_verifier = AsymmetricSignatureVerifier(jwks_url)
    token_verifier = TokenVerifier(
        signature_verifier=signature_verifier, issuer=issuer, audience=AUTH0_CLIENT_ID
    )
    token_verifier.verify(id_token)
