from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

filters = Extension(name = 'median_filter.medfilt', 
                    include_dirs=[np.get_include()],
                    sources = ['median_filter/medfilt.pyx'])

extensions = [
    filters
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    )
)    

# copy the .so file to the correct location
#os.system(f"cp build/lib.linux-x86_64-cpython-311/modules/utils/bin_funcs.cpython-311-x86_64-linux-gnu.so .")