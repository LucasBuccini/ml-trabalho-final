import sys
print("Python:", sys.version)
try:
    import sklearn; print("sklearn:", sklearn.__version__)
except: print("sklearn: NOT FOUND")
try:
    import pygam; print("pygam ok")
except: print("pygam: NOT FOUND")
try:
    import nbformat; print("nbformat:", nbformat.__version__)
except: print("nbformat: NOT FOUND")
try:
    import nbconvert; print("nbconvert ok")
except: print("nbconvert: NOT FOUND")
