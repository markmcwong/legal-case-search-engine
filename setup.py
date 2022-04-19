import subprocess

def setup_dependencies():
    try:
        import smart_open
    except ModuleNotFoundError:
        subprocess.call(
            "cd smart_open && python3 setup.py install --user", shell=True
        )

    try:
        import cython
    except ModuleNotFoundError:
        subprocess.call(
            "cd cython && python3 setup.py install --user", shell=True
        )

    try:
        import numpy
    except ModuleNotFoundError:
        subprocess.call(
            "cd numpy && python3 setup.py install --user", shell=True
        )
    try:
        import gensim
    except ModuleNotFoundError:
        subprocess.call(
            "cd gensim && python3 setup.py install --user", shell=True
        )