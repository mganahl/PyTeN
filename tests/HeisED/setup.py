from distutils.core import setup,Extension
from Cython.Build import cythonize
extensions=[Extension("XXZED",["XXZED.pyx"],language="c++")]
setup(ext_modules=cythonize(extensions))
