"""Evaluation-only library for the MODA General Attribute Suite.

Contains scoring, normalisation and parsing. Deliberately free of model,
training and serving code so a scorer can be run by anyone.
"""

from .conditional import normalize_value  # noqa: F401
from .entries import attributes_from_entry  # noqa: F401
from .image_attributes import evaluate_image_attributes  # noqa: F401
