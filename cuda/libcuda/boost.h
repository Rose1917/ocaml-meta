#ifndef BOOST_H
#define BOOST_H
void mul_avx(double *A,double *B,double *C,int m,int p,int n);
void mul_fma(double *A,double *B,double *C,int m,int p,int n);
void mul_omp(double *A,double *B,double *C,int m,int p,int n);

#endif
