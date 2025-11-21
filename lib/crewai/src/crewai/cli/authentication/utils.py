import jwt
from jwt import PyJWKClient


def validate_jwt_token(
    jwt_token: str, jwks_url: str, issuer: str, audience: str
) -> dict:
    """
    Verify the token's signature and claims using PyJWT.
    :param jwt_token: The JWT (JWS) string to validate.
    :param jwks_url: The URL of the JWKS endpoint.
    :param issuer: The expected issuer of the token.
    :param audience: The expected audience of the token.
    :return: The decoded token.
    :raises Exception: If the token is invalid for any reason (e.g., signature mismatch,
                       expired, incorrect issuer/audience, JWKS fetching error,
                       missing required claims).
    """

    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(jwt_token)

        _unverified_decoded_token = jwt.decode(
            jwt_token, options={"verify_signature": False}
        )
        return jwt.decode(
            jwt_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            leeway=10.0,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )

    except jwt.ExpiredSignatureError as e:
        raise Exception("Token has expired.") from e
    except jwt.InvalidAudienceError as e:
        actual_audience = _unverified_decoded_token.get("aud", "[no audience found]")
        raise Exception(
            f"Invalid token audience. Got: '{actual_audience}'. Expected: '{audience}'"
        ) from e
    except jwt.InvalidIssuerError as e:
        actual_issuer = _unverified_decoded_token.get("iss", "[no issuer found]")
        raise Exception(
            f"Invalid token issuer. Got: '{actual_issuer}'. Expected: '{issuer}'"
        ) from e
    except jwt.MissingRequiredClaimError as e:
        raise Exception(f"Token is missing required claims: {e!s}") from e
    except jwt.exceptions.PyJWKClientError as e:
        raise Exception(f"JWKS or key processing error: {e!s}") from e
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid token: {e!s}") from e
