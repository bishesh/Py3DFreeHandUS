Installation
============

Move the folder Py3DFreeHandUS into *Python folder*/Lib/site-packages. If the you can run the following in the Python shell::

	from Py3DFreeHandUS import process

without any error, then you're done!

If you receive the warning *Module calib_c was not found* on ``from calib_c import *``, then you strongly suggested to *cythonize* some functions that would require much more time if implemented in pure Python. To do this:

1) Make sure to have Cython + gcc installed (this one is available after installing `MinGW <http://www.mingw.org/>`_ on Windows)
2) Open a console/terminal
3) Change directory to the Py3DFreeHandUS folder
4) Type: *python cython_setup.py build_ext --inplace --compiler=mingw32*
5) Cython should automatically generate the following .pyd modules in the folder: calib_c, image_utils_c


In case cythonization is not possible, the following function calls will not be available and generate errors:

- ``process.Process.calibrateProbe(... , method='maximize_NCCfast', ...)``


