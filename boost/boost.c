#include <stdio.h>
#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <string.h>
#include "common.h"

//#define DEBUG

/* this file implement three kinds of mat-mul boost operation
 * 1) avx boost
 * 3) fma boost 
 */

//transpose: M (n x p) -> A^(T)(p x n)
void T_Matrix(double *M,double *M_T,int n,int p)
{
    #ifdef DEBUG
	printf("in T_Matrix\n");
    #endif
    for(int i = 0;i<n;i++)
    {
        for(int j = 0;j<p;j++)
        {
            M_T[j*n+i] = M[i*p+j];
        }
    }
    #ifdef DEBUG
	printf("in T_Matrix:end\n");
    #endif
}

//the matrix implementation based on the axv vector instruction
//A(m x n) dot B(n x p) -> C(m x p)
void mul_avx(double *A,double *B,double *C,int m,int p,int n)
{
    #ifdef DEBUG
	printf("in mul avx init\n");
    #endif
    double *B_T;
    B_T  = (double*)malloc(p*n*sizeof(double*));

    T_Matrix(B,B_T,n,p);//得到B的转置矩阵
    #ifdef DEBUG
	printf("after transpose\n");
    #endif

    const int n_reduced_4 = n - n % 4;
    __m256d op0,op1,tgt,tmp_vec;
    double dvec[4];
    for(int i = 0;i < m;i++)
    {
        for(int k = 0;k<p;k++)
        {
            double res = 0;
            tgt = _mm256_setzero_pd();
            for(int j = 0;j<n_reduced_4;j += 4)
            {
                op0 = _mm256_loadu_pd(&A[i*n+j]);
                op1 = _mm256_loadu_pd(&B_T[k*p+j]);
                tmp_vec = _mm256_mul_pd(op0,op1);
                tgt = _mm256_add_pd(tmp_vec,tgt);
            }
            _mm256_storeu_pd(dvec,tgt);
            for(int l = 0;l<4;l++)
            {
                res += dvec[l];
            }
            for(int l = n_reduced_4;l<n;l++)
            {
                res += A[i*n+l]*B_T[k*n+l];
            }
            C[i*p+k] = res;
        }
        
    }
    #ifdef DEBUG
	printf("in mul avx before free\n");
    #endif

    free(B_T);
    #ifdef DEBUG
	printf("in mul avx after free\n");
    #endif
}

//the matrix mul operation based on the fma vector instruction
//m x n X n x p -> m x p
void mul_fma(double *A,double *B,double *C,int m,int p,int n)
{
    /*malloc for B transpose */
    double *B_T;
    B_T  = (double*)malloc(n*p*sizeof(double*));

    T_Matrix(B,B_T,n,p);

    const int n_reduced_4 = n - n % 4;
    __m256d op1,op0,tgt,tmp_vec;
    double dvec[4];
    for(int i = 0;i < m;i++)
    {
        for(int k = 0;k<p;k++)
        {
            double res = 0;
            tgt = _mm256_setzero_pd();
            for(int j = 0;j<n_reduced_4;j += 4)
            {
                op0 = _mm256_loadu_pd(&A[i*n+j]);
                op1 = _mm256_loadu_pd(&B_T[k*n+j]);
                tgt = _mm256_fmadd_pd(op0,op1,tgt);
            }
            _mm256_storeu_pd(dvec,tgt);
            for(int l = 0;l<4;l++)
            {
                res += dvec[l];
            }
            for(int l = n_reduced_4;l<n;l++)
            {
                res += A[i*n+l]*B_T[k*n+l];
            }
            C[i*p+k] = res;
        }
        
    }
    free(B_T);
}

//mul implementation by the multi-processors
void mul_omp(double *A,double *B,double *C,int m,int p,int n)
{
    
    #pragma omp parallel num_threads(8)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
		    C[j*p+k] += A[j*n+i] * B[i*p + k];
            }
        }
    }
}

