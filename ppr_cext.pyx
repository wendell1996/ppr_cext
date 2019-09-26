import numpy
cimport numpy as np
cimport cython
import scipy.sparse as sp

np.import_array()

cdef extern from "ppr.h" nogil:
    void top_k_dot[I,T](const I n_row,
                        const I n_col,
                        const I k,
                        const I Ap[],
                        const I Aj[],
                        const T Ax[],
                        const I Bp[],
                        const I Bj[],
                        const T Bx[],
                              I Cp[],
                              I Cj[],
                              T Cx[]);
    
    void count_nnz[I](const I n_row,
                      const I n_col,
                      const I Ap[],
                      const I Aj[],
                      const I Bp[],
                      const I Bj[],
                            I Cp[]);
    
    void count_nnz_parallel[I](const I   n_row,
                               const I   n_col,
                               const I   Ap[],
                               const I   Aj[],
                               const I   Bp[],
                               const I   Bj[],
                                     I   Cp[],
                               const int num_jobs);
    
    void dot_parallel[I,T](const I   n_row,
                           const I   n_col,
                           const I   Ap[],
                           const I   Aj[],
                           const T   Ax[],
                           const I   Bp[],
                           const I   Bj[],
                           const T   Bx[],
                                 I   Cp[],
                                 I   Cj[],
                                 T   Cx[],
                           const int num_jobs);

    void squeeze_k_parallel[I,T](const I   n_row,
                                 const I   k,
                                       I   Cp[],
                                       I   Cj[],
                                       T   Cx[],
                                 const int num_jobs);


@cython.boundscheck(False)
@cython.wraparound(False)
def _ppr_dot(n_row,
             n_col,
             k,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Ap,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Aj,
             np.ndarray[np.float32_t,  ndim=1, mode="c"] Ax,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Bp,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Bj,
             np.ndarray[np.float32_t,  ndim=1, mode="c"] Bx,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Cp,
             np.ndarray[np.int32_t,    ndim=1, mode="c"] Cj,
             np.ndarray[np.float32_t,  ndim=1, mode="c"] Cx):
    top_k_dot(<int>     n_row,
              <int>     n_col,
              <int>     k,
              <int*>    np.PyArray_DATA(Ap),
              <int*>    np.PyArray_DATA(Aj),
              <float*>  np.PyArray_DATA(Ax),
              <int*>    np.PyArray_DATA(Bp),
              <int*>    np.PyArray_DATA(Bj),
              <float*>  np.PyArray_DATA(Bx),
              <int*>    np.PyArray_DATA(Cp),
              <int*>    np.PyArray_DATA(Cj),
              <float*>  np.PyArray_DATA(Cx))

@cython.boundscheck(False)
@cython.wraparound(False)
def _count_nnz(n_row,
               n_col,
               np.ndarray[np.int32_t,    ndim=1, mode="c"] Ap,
               np.ndarray[np.int32_t,    ndim=1, mode="c"] Aj,
               np.ndarray[np.int32_t,    ndim=1, mode="c"] Bp,
               np.ndarray[np.int32_t,    ndim=1, mode="c"] Bj,
               np.ndarray[np.int32_t,    ndim=1, mode="c"] Cp):
    count_nnz(<int>  n_row,
              <int>  n_col,
              <int*> np.PyArray_DATA(Ap), 
              <int*> np.PyArray_DATA(Aj),
              <int*> np.PyArray_DATA(Bp),
              <int*> np.PyArray_DATA(Bj),
              <int*> np.PyArray_DATA(Cp))

@cython.boundscheck(False)
@cython.wraparound(False)
def _count_nnz_parallel(n_row,
                        n_col,
                        np.ndarray[np.int32_t,    ndim=1, mode="c"] Ap,
                        np.ndarray[np.int32_t,    ndim=1, mode="c"] Aj,
                        np.ndarray[np.int32_t,    ndim=1, mode="c"] Bp,
                        np.ndarray[np.int32_t,    ndim=1, mode="c"] Bj,
                        np.ndarray[np.int32_t,    ndim=1, mode="c"] Cp,
                        num_jobs):
    count_nnz_parallel(<int>  n_row,
                       <int>  n_col,
                       <int*> np.PyArray_DATA(Ap), 
                       <int*> np.PyArray_DATA(Aj),
                       <int*> np.PyArray_DATA(Bp),
                       <int*> np.PyArray_DATA(Bj),
                       <int*> np.PyArray_DATA(Cp),
                       <int>  num_jobs)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _dot_parallel(n_row,
                    n_col,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Ap,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Aj,
                    np.ndarray[np.float32_t,  ndim=1, mode="c"] Ax,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Bp,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Bj,
                    np.ndarray[np.float32_t,  ndim=1, mode="c"] Bx,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Cp,
                    np.ndarray[np.int32_t,    ndim=1, mode="c"] Cj,
                    np.ndarray[np.float32_t,  ndim=1, mode="c"] Cx,
                    num_jobs):
    dot_parallel(<int>     n_row,
                 <int>     n_col,
                 <int*>    np.PyArray_DATA(Ap),
                 <int*>    np.PyArray_DATA(Aj),
                 <float*>  np.PyArray_DATA(Ax),
                 <int*>    np.PyArray_DATA(Bp),
                 <int*>    np.PyArray_DATA(Bj),
                 <float*>  np.PyArray_DATA(Bx),
                 <int*>    np.PyArray_DATA(Cp),
                 <int*>    np.PyArray_DATA(Cj),
                 <float*>  np.PyArray_DATA(Cx),
                 <int>     num_jobs)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _squeeze_k_parallel(n_row,
                          k,
                          np.ndarray[np.int32_t,   ndim=1] Cp,
                          np.ndarray[np.int32_t,   ndim=1] Cj,
                          np.ndarray[np.float32_t, ndim=1] Cx,
                          num_jobs):
    squeeze_k_parallel(<int>     n_row,
                       <int>     k,
                       <int*>    np.PyArray_DATA(Cp),
                       <int*>    np.PyArray_DATA(Cj),
                       <float* > np.PyArray_DATA(Cx),
                       <int>     num_jobs)

@cython.boundscheck(False)
@cython.wraparound(False)
def ppr_dot(a,b,k):
    n_row = a.shape[0]
    n_col = b.shape[1]
    a.indptr  = numpy.asarray(a.indptr,  dtype = numpy.int32)
    a.indices = numpy.asarray(a.indices, dtype = numpy.int32)
    b.indptr  = numpy.asarray(b.indptr,  dtype = numpy.int32)
    b.indices = numpy.asarray(b.indices, dtype = numpy.int32)
    nnz = n_row * k
    Cp = numpy.empty(n_row+1, dtype = numpy.int32)
    Cj = numpy.empty(nnz,     dtype = numpy.int32)
    Cx = numpy.zeros(nnz,     dtype = numpy.float32)
    _ppr_dot(n_row,
             n_col,
             k,
             a.indptr,
             a.indices,
             a.data,
             b.indptr,
             b.indices,
             b.data,
             Cp,
             Cj,
             Cx)
    return sp.csr_matrix((Cx,Cj,Cp),shape=(n_row,n_col),dtype=numpy.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef squeeze(a,k,num_jobs=-1):
    n_row = a.shape[0]
    _squeeze_k_parallel(n_row,
                        k,
                        a.indptr,
                        a.indices,
                        a.data,
                        num_jobs)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dot(a,b,num_jobs=-1):
    n_row = a.shape[0]
    n_col = b.shape[1]
    Cp = numpy.empty(n_row+1, dtype = numpy.int32)
    _count_nnz_parallel(n_row, 
                        n_col,
                        a.indptr,
                        a.indices,
                        b.indptr,
                        b.indices,
                        Cp,
                        num_jobs)
    nnz = Cp[n_row]
    Cj = numpy.empty(nnz, dtype = numpy.int32)
    Cx = numpy.zeros(nnz, dtype = numpy.float32)
    _dot_parallel(n_row,
                  n_col,
                  a.indptr,
                  a.indices,
                  a.data,
                  b.indptr,
                  b.indices,
                  b.data,
                  Cp,
                  Cj,
                  Cx,
                  num_jobs)
    return sp.csr_matrix((Cx,Cj,Cp),shape=(n_row,n_col),dtype=numpy.float32)

