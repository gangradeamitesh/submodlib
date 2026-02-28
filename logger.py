import logger

_logger = logger.get_logger("submodlib")
_logger.setLevel(logger.INFO)

if not _logger.handlers:
    handler = logger.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

_logger.propagate = False  
_logger.disabled = True

def get_logger():
    return _logger

def enable_logging():
    _logger.setLevel(logging.INFO)
    _logger.disabled = False

def disable_logging():
    _logger.disabled = True 
               