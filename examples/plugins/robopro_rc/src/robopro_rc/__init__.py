"""RC-series arms of Robopro (formerly Rozum Robotics) for the platform.

An adapter of the command class over the vendor's ``RobotApi`` (API 1.4.1), a
recording channel to check it without the robot, and the forward kinematics the
hand-written RC5 description is checked with on the robot (ADR-0055).

Importing this package imports neither the SDK nor numpy: the SDK is loaded when a
link is opened, in the one module that needs it (``sdk``).
"""
