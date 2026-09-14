"""Unitree G1 for the platform: a DDS adapter for every environment, and its twins.

Importing this package imports neither CycloneDDS nor the vendor's SDK. Both
are loaded when a link is opened, in the one module that needs them
(``cyclone``), so the platform can list, validate and check the environment of a
robot on a machine where they are absent.
"""
