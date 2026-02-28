import logging

_logging = logging.getLogger("submodlib")
_logging.setLevel(logging.INFO)

if not _logging.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    _logging.addHandler(handler)

_logging.propagate = False  
_logging.disabled = True

def get_logging():
    return _logging

def enable_logging():
    _logging.setLevel(logging.INFO)
    _logging.disabled = False

def disable_logging():
    _logging.disabled = True 
               