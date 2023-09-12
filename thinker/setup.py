from setuptools import setup
from Cython.Build import cythonize


setup(ext_modules=cythonize("thinker/cenv.pyx", include_path=[]))
setup(ext_modules=cythonize("thinker/cenv_simple.pyx", include_path=[]))
