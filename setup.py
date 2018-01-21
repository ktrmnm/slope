from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

numpy_include = numpy.get_include()

setup(
    name = "slope",
    author = "Kentaro Minami",
    ext_modules = cythonize(
        Extension("slope._prox",
            sources=["slope/_prox.pyx"],
            include_dirs = [numpy_include]
        )
    ),
    cmdclass = {'build_ext': build_ext}
)
