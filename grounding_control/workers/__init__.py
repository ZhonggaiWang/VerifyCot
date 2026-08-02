"""Long-lived out-of-process workers for verifier and expert models.

Worker modules are intentionally not imported here.  Keeping this package
initializer empty avoids constructing worker dependencies twice when a worker
is launched with ``python -m``.
"""
