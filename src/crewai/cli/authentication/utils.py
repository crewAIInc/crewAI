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

    decoded_token = None

    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(jwt_token)

        _unverified_decoded_token = jwt.decode(
            jwt_token, options={"verify_signature": False}
        )
        decoded_token = jwt.decode(
            jwt_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )
        return decoded_token

    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired.")
    except jwt.InvalidAudienceError:
        actual_audience = _unverified_decoded_token.get("aud", "[no audience found]")
        raise Exception(
            f"Invalid token audience. Got: '{actual_audience}'. Expected: '{audience}'"
        )
    except jwt.InvalidIssuerError:
        actual_issuer = _unverified_decoded_token.get("iss", "[no issuer found]")
        raise Exception(
            f"Invalid token issuer. Got: '{actual_issuer}'. Expected: '{issuer}'"
        )
    except jwt.MissingRequiredClaimError as e:
        raise Exception(f"Token is missing required claims: {str(e)}")
    except jwt.exceptions.PyJWKClientError as e:
        raise Exception(f"JWKS or key processing error: {str(e)}")
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid token: {str(e)}")
