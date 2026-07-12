class VectorStoreError(Exception):
    """Base exception for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector-store configuration is invalid."""


class VectorStoreDataError(VectorStoreError):
    """Raised when vector-store input data is inconsistent."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector database cannot be reached."""
