from contextlib import contextmanager
from contextvars import ContextVar

is_recomputing = ContextVar("is_recomputing", default=False)

@contextmanager
def recompute_context():
    token = is_recomputing.set(True)
    try:
        yield
    finally:
        is_recomputing.reset(token)
