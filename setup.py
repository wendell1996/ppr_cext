from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import numpy as np

ext_modules = [
    Extension("ppr_cext",
              sources = ["ppr_cext.pyx"],
              include_dirs = [np.get_include(),
                             os.getcwd()],
              language = "c++",
              library_dirs = ["."],
              libraries = ["ppr"],
              extra_compile_args=["-std=c++11"]
    )
]

setup(
    name = "ppr_cext",
    version = "0.0.1",
    author = "Chen Huidi",
    description = "Personal PageRank(C++ extension)",
    ext_modules = cythonize(ext_modules)
)
