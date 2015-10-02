#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    name="projection_fcube",
    version="0.1",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("anchor_new",
                             sources=["anchor_new.pyx", "c_anchor.c"],
                             include_dirs=[numpy.get_include()],library_dirs=['/usr/local/lib'],libraries=['wht','gsl', 'cblas','m'])],
)

