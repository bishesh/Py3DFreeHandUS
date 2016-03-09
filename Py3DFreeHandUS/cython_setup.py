
import os
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [
            Extension('image_utils_c', ['image_utils_c.pyx']),
            Extension('calib_c', ['calib_c.pyx'])
            
            ]

#ext_modules = [Extension("*", "*.pyx")]

setup(cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize(ext_modules, annotate=True),
      include_dirs=[numpy.get_include(),
                    os.path.join(numpy.get_include(), 'numpy')]
    )