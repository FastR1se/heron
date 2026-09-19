"""The command line as a module: how the platform starts a process of its own (ADR-P8D).

``python -m`` of this package runs the same commands as the installed script, with
the interpreter that started it -- so a robot's process is this interpreter and
this package, whatever the script on ``PATH`` points at.
"""

from .main import app

if __name__ == "__main__":
    app()
