import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Python interface
setup(
    name='arcsim',
    install_requires=['torch'],
    ext_modules=[
        CppExtension(
            name='arcsim',
#            include_dirs=['./arcsim-ori/src/','./arcsim/dependencies/include'],
            include_dirs=['./arcsim/src/','./arcsim/dependencies/include'],
            sources=[
#                'pybind/bind-ori.cpp',
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch','json','taucs','alglib',
            #'png','z','lapack','blas','boost_system-mt','boost_filesystem-mt','boost_thread-mt','gomp','glut','GLU','GL','glapi','GLdispatch'],
            'png','z','lapack','blas','boost_system','boost_filesystem','boost_thread','gomp','glut','GLU','GL'],
            library_dirs=['objs','./arcsim/dependencies/lib'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    url='https://github.com/chrischoy/MakePytorchPlusPlus',
    zip_safe=False,
)
