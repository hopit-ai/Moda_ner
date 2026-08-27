"""Model architecture and per-route inference for the published MODA_NER(V) weights.

Published so that the weights are loadable. This is the architecture, not the
training recipe: there is no training loop, no losses, no calibration fitting and
no data pipeline here.
"""

from .routes import ROUTES, CatalogBackend, CropBackend, FullbodyBackend  # noqa: F401
