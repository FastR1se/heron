"""What a robot package tests itself with (ADR-0056).

Imported by tests alone, never by the platform at run time: it needs ``pytest``,
which the core does not install.
"""
