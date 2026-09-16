"""Training: deriving an environment from a contract, and running one process.

The backend lives in the core, like the simulation backends and unlike anything
vendor-specific, because it knows no robot and no task: what it builds comes from
the contract, and what is being learned arrives through the sixth extension point
(ADR-0044). Isaac, torch and the learning library are imported in the training
process alone and are not dependencies of the platform.
"""
