"""PQ-DNA: firmware-resident lifetime power-quality stress signatures.

A reference implementation of a compact, NVM-resident power-quality stress
signature and per-subsystem Component Health Score (CHS) for power-electronics
edge devices. Written in NumPy for development and validation; the on-device
path exports to fixed-point C via `pqdna.edge_export`.

See `docs/METHOD.md` for the method description and the limits of the
evaluation.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

__version__ = "0.1.0"

from .chs import FAILURE_MODES, ComponentHealthScorer
from .features import PQFeatureExtractor, PQFeatures
from .pqdna import PQDNAAggregator, PQDNAState
from .rul import RULEstimator
from .synth import MeterEnvironment, SyntheticMeterFleet

__all__ = [
    "PQFeatureExtractor",
    "PQFeatures",
    "PQDNAState",
    "PQDNAAggregator",
    "ComponentHealthScorer",
    "FAILURE_MODES",
    "RULEstimator",
    "SyntheticMeterFleet",
    "MeterEnvironment",
]
