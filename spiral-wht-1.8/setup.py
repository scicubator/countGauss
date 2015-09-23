# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("anchor", ["anchor.pyx", "anchor_fast.cpp"], language="c++", library_dirs=['/usr/local/lib'],libraries=['wht'])],
cmdclass = {'build_ext': build_ext})

