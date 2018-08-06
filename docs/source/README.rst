mpi4py-fft
----------


.. image:: https://readthedocs.org/projects/mpi4py-fft/badge/?version=latest
   :target: https://mpi4py-fft.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


mpi4py-fft is a Python package for computing Fast Fourier Transforms (FFTs). 
Large arrays are distributed and communications are handled under the hood by 
MPI for Python (mpi4py). To distribute large arrays we are using a 
`new and completely generic algorithm <https://arxiv.org/abs/1804.09536>`_
that allows for any index set of a multidimensional array to be distributed. We 
can distribute just one index (a slab decomposition), two index sets (pencil 
decomposition) or even more for higher-dimensional arrays.

In mpi4py-fft there is also included a Python interface to the 
`FFTW <http://www.fftw.org>`_ library. This interface can be used without MPI, 
much like `pyfftw <https://hgomersall.github.io/pyFFTW/>`_, and even for 
real-to-real transforms, like discrete cosine or sine transforms.

