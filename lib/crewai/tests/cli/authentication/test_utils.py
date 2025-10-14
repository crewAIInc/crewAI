import unittest
from unittest.mock import MagicMock, patch

import jwt

from crewai.cli.authentication.utils import validate_jwt_token


@patch("crewai.cli.authentication.utils.PyJWKClient", return_value=MagicMock())
@patch("crewai.cli.authentication.utils.jwt")
class TestUtils(unittest.TestCase):
    def test_validate_jwt_token(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.return_value = {"exp": 1719859200}

        # Create signing key object mock with a .key attribute
        mock_pyjwkclient.return_value.get_signing_key_from_jwt.return_value = MagicMock(
            key="mock_signing_key"
        )

        jwt_token = "aaaaa.bbbbbb.cccccc"  # noqa: S105

        decoded_token = validate_jwt_token(
            jwt_token=jwt_token,
            jwks_url="https://mock_jwks_url",
            issuer="https://mock_issuer",
            audience="app_id_xxxx",
        )

        mock_jwt.decode.assert_called_with(
            jwt_token,
            "mock_signing_key",
            algorithms=["RS256"],
            audience="app_id_xxxx",
            issuer="https://mock_issuer",
            leeway=10.0,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )
        mock_pyjwkclient.assert_called_once_with("https://mock_jwks_url")
        self.assertEqual(decoded_token, {"exp": 1719859200})

    def test_validate_jwt_token_expired(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.ExpiredSignatureError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_audience(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidAudienceError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_issuer(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidIssuerError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_missing_required_claims(
        self, mock_jwt, mock_pyjwkclient
    ):
        mock_jwt.decode.side_effect = jwt.MissingRequiredClaimError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_jwks_error(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.exceptions.PyJWKClientError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_token(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidTokenError
        with self.assertRaises(Exception):  # noqa: B017
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",  # noqa: S106
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )
