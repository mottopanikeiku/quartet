"""Expected quartet domain and trust-boundary failures."""

from __future__ import annotations


class QuartetError(Exception):
    """Base class for failures safe to present to a command-line user."""


class SchemaError(QuartetError, ValueError):
    """A decoded value does not satisfy a quartet schema."""


class ArtifactError(QuartetError):
    """An artifact could not be read, verified, or committed safely."""


class ProtocolError(QuartetError, ValueError):
    """A requested collection or replay violates its frozen protocol."""
