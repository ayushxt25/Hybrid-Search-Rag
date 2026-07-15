import hashlib
import hmac
import re

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


class ApiKeyAuthenticator:
    def __init__(self, expected_sha256: str) -> None:
        if not _SHA256_HEX.fullmatch(expected_sha256):
            raise ValueError("expected_sha256 must be a SHA-256 hex digest.")

        self.expected_sha256 = expected_sha256.lower()

    def authenticate(self, provided_key: str | None) -> bool:
        if provided_key is None or not provided_key.strip():
            return False

        digest = hashlib.sha256(provided_key.encode()).hexdigest()
        return hmac.compare_digest(digest, self.expected_sha256)
