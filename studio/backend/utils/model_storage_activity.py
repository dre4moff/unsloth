# SPDX-License-Identifier: AGPL-3.0-only
"""Exclude model-file relocation from new media loads and training starts."""

import functools
import threading
from contextlib import contextmanager

from fastapi import HTTPException

_lock = threading.RLock()
_readers = 0
_moving = False


@contextmanager
def relocation_reservation():
    global _moving
    with _lock:
        if _moving or _readers:
            raise HTTPException(
                409, "Wait for the current model load or training start before moving models."
            )
        _moving = True
    try:
        yield
    finally:
        with _lock:
            _moving = False


def model_file_operation(function):
    """Guard a synchronous loader, including worker-thread entry points."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        global _readers
        with _lock:
            if _moving:
                raise HTTPException(
                    409, "Wait for the model move to finish before loading or training models."
                )
            _readers += 1
        try:
            return function(*args, **kwargs)
        finally:
            with _lock:
                _readers -= 1

    return wrapped
