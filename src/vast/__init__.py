"""Public facade for the interim VAST research snapshot."""

from .method import describe_method, load_default_config

__version__ = "0.1.0.dev0"

__all__ = ("__version__", "describe_method", "load_default_config")
