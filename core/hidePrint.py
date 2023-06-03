import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# Decorator to hide prints
def hidePrints(func):
    def wrapper(*args, **kwargs):
        with HiddenPrints():
            return func(*args, **kwargs)
    return wrapper
