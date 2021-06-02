#include <stdio.h>
#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include "common.h"
#include "boost.h"
#include "mat_cuda.h"

#define BUFFER_SIZE 1000000
//#define DEBUG



double res[BUFFER_SIZE];

__global__ void mul_cuda(double* x, double* y, double* z, int m, int n, int p)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    double tmp = 0.0;
        for (int i = 0; i < n; i++)
        {
            tmp += x[row * n + i] * y[col + i * p];
        }
        z[row * p + col] = tmp;
}

//matrix operation
CAMLprim value 
cuda_mat_mul (value x,value y){

#ifdef DEBUG
	printf("now in the cuda_mul function\n");
#endif
	
	//obtain the data from value type
	double* x_val =(double*) Caml_ba_data_val(x);
	double* y_val =(double*) Caml_ba_data_val(y);

	//obtain the properties pointer from the value type
	struct caml_ba_array *x_pro =Caml_ba_array_val(x);
	struct caml_ba_array *y_pro =Caml_ba_array_val(y);
	
	//check the shape of the two matrixs
	int x_dimension = x_pro -> num_dims;
	int y_dimension = y_pro -> num_dims;

	if( x_dimension != 2 || y_dimension != 2){
		printf("cuda_mat_mul:dimension error \n");
		printf("left operand dimension number: %d,the right dimension number: %d\n",x_dimension,y_dimension);
		exit(DIMENSION_NOT_QUALIFIED);
	}

#ifdef DEBUG
	printf("first judgement\n");
#endif
	//define the shape of the two matrix
	int x_r,x_c;
	int y_r,y_c;

	//obtain the row and col of x and y
	x_r = x_pro -> dim[0];
	x_c = x_pro -> dim[1];

	y_r = y_pro -> dim[0];
	y_c = y_pro -> dim[1];

	//check if the shape matchs
	if( x_c != y_r ){
		printf("cuda_mat_mul:shape not match\n");
		printf("the shape of left operand %d x %d",x_r,x_c);
		printf("the shape of right operand %d x %d",y_r,y_c);
		exit(SHAPE_NOT_MATCH);
	}	

#ifdef DEBUG
	//traverse the x
	for ( int i = 0; i<x_r ; i++){
		for (int j = 0;j<x_c;j++)
			printf(" %f",x_val[i*x_c+j]);
		printf ("\n");
	}

	//traverse the y
	for ( int i = 0; i<y_r ; i++){
		for (int j = 0;j<y_c;j++)
			printf(" %f",y_val[i*y_c+j]);
		printf ("\n");
	}
#endif

	//so the result matrix shape is x_r * y_c
	long  res_r = x_r;
	long  res_c = y_c;

	//prepare for the result buffer
	if(res_r * res_c > BUFFER_SIZE) {
		printf ("cuda mat_mul:buffer is not enough");
		exit(BUFFER_OVERFLOW);
	}
	
	//init the matrix c
	memset(res,0,sizeof(double) * res_r * res_c);

	//the GPU memory pointer	
	double *x_cuda, *y_cuda, *res_cuda;

	//malloc the memory from gpu
	cudaMalloc((void**)&x_cuda, sizeof(double) * x_r * x_c);
	cudaMalloc((void**)&y_cuda, sizeof(double) * y_r * y_c);
	cudaMalloc((void**)&res_cuda, sizeof(double) * res_r * res_c);

	//set the value
	cudaMemcpy(x_cuda, x_val, sizeof(double) * x_r * x_c, cudaMemcpyHostToDevice);
	cudaMemcpy(y_cuda, y_val, sizeof(double) * y_r * y_c, cudaMemcpyHostToDevice);
	cudaMemcpy(res_cuda, res, sizeof(double) * res_r * res_c, cudaMemcpyHostToDevice);
	
	//we can not know the row and column here
	//so we transfer the two dimensional result matrix to a liner
	dim3 grid (res_c);
	dim3 block(res_r);

	mul_cuda<<<grid, block>>>(x_cuda, y_cuda, res_cuda, x_r, x_c, y_c);

	//copy back the data
	cudaMemcpy(res, res_cuda, sizeof(double) * res_r * res_c, cudaMemcpyDeviceToHost);

	//all the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc_dims(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,2,res,res_r,res_c);
}

int main(){
	printf("in mian");
	return 0;
}
