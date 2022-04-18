import subprocess

def setup_dependencies():
    subprocess.call(
        "cd smart_open && python3 setup.py install --user", shell=True
    )

    subprocess.call(
        "cd numpy && python3 setup.py install --user", shell=True
    )

    subprocess.call(
        "cd gensim && python3 setup.py install --user", shell=True
    )