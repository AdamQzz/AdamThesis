import os
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SOURCE_DIR, 'data')
RESULT_DIR = os.path.join(SOURCE_DIR, 'result')

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

mkdir(DATA_DIR)
mkdir(RESULT_DIR)