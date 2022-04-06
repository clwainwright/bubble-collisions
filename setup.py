from distutils.core import setup, Extension
#from Cython.Build import cythonize
#from Cython.Distutils import build_ext
import numpy as np

module1 = Extension(
    'bubble_collisions.adaptiveGridTest',
    define_macros=[
        ('PY_ARRAY_UNIQUE_SYMBOL', 'adaptiveGridTest_unique_pysymb'),
    #    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ],
    sources = ['src/adaptiveGrid.c', 'src/adaptiveGridTesting.c'],
    include_dirs = [np.get_include()+"/numpy", "./src"],
    extra_compile_args=['-Wno-shorten-64-to-32']
    )

simulation_module = Extension(
    'bubble_collisions.simulation',
    define_macros=[
        ('PY_ARRAY_UNIQUE_SYMBOL', 'bubble_collisions_simulation_unique_pysymb'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ],
    sources = [
        'src/simulation.c','src/adaptiveGrid.c', 
        'src/bubbleEvolution.c', 'src/gridInterpolation.c'],
    include_dirs = [np.get_include()+"/numpy", "./src"],
    extra_compile_args=['-Wno-shorten-64-to-32']
    )

models_module = Extension(
    'bubble_collisions.models',
    define_macros=[
        ('PY_ARRAY_UNIQUE_SYMBOL', 'bubble_collisions_models_unique_pysymb'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
    ],
    sources = ['src/models.c'],
    include_dirs = [np.get_include()+"/numpy", "./src"],
    extra_compile_args=['-Wno-shorten-64-to-32']
    )

setup(
    name = 'bubble_collisions',
    version = '2.0',
    description = "1D Grid adaptation for bubble collisions in GR",
#   cmdclass = {'build_ext': build_ext},
    ext_modules = [simulation_module, models_module],
#    ext_modules = [models_module],
    packages=['bubble_collisions', 'bubble_collisions.cosmoTransitions'])