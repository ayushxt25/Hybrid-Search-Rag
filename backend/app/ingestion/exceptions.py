class DocumentLoadingError(Exception):
    """Base exception for document-loading failures."""


class DocumentNotFoundError(DocumentLoadingError):
    """Raised when the requested document does not exist."""


class UnsupportedFileTypeError(DocumentLoadingError):
    """Raised when the document format is not supported."""


class DocumentTooLargeError(DocumentLoadingError):
    """Raised when the document exceeds the configured size limit."""


class EmptyDocumentError(DocumentLoadingError):
    """Raised when the document contains no usable text."""


class DocumentDecodingError(DocumentLoadingError):
    """Raised when document bytes cannot be decoded safely."""


class CorruptedDocumentError(DocumentLoadingError):
    """Raised when a document cannot be parsed safely."""


class EncryptedDocumentError(DocumentLoadingError):
    """Raised when a document requires a password."""


class NoExtractableTextError(DocumentLoadingError):
    """Raised when a document contains no machine-readable text."""
