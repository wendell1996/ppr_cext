#ifndef __PPR_H__
#define __PPR_H__

#include <vector>
#include <limits.h>
#include <stdexcept>
#include <queue>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <ctime>

#define NPY_MAX_INTP INT_MAX
#define npy_intp int

template <class I,class T>
void top_k_dot(const I n_row,
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

template <class I>
void count_nnz(const I n_row,
               const I n_col,
               const I Ap[],
               const I Aj[],
               const I Bp[],
               const I Bj[],
                     I Cp[]);

template <class I>
void count_nnz_parallel(const I   n_row,
                        const I   n_col,
                        const I   Ap[],
                        const I   Aj[],
                        const I   Bp[],
                        const I   Bj[],
                              I   Cp[],
                        const int num_jobs);

template <class I, class T>
void dot_parallel(const I   n_row,
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

template <class I, class T>
void dot(const I n_row,
         const I n_col,
         const I Ap[],
         const I Aj[],
         const T Ax[],
         const I Bp[],
         const I Bj[],
         const T Bx[],
               I Cp[],
               I Cj[],
               T Cx[])
{
    dot_parallel(n_row,
                 n_col,
                 Ap,
                 Aj,
                 Ax,
                 Bp,
                 Bj,
                 Bx,
                 Cp,
                 Cj,
                 Cx,
                 -1);
}

template <class I,class T>
void squeeze_k_parallel(const I   n_row,
                        const I   k,
                              I   Cp[],
                              I   Cj[],
                              T   Cx[],
                        const int num_jobs);

#endif
