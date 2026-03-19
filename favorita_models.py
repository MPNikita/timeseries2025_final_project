"""Root compatibility wrapper for legacy imports."""

from src.favorita import models as _legacy

_exported = [name for name in dir(_legacy) if not name.startswith("__")]
globals().update({name: getattr(_legacy, name) for name in _exported})
__all__ = _exported
