#include <stdio.h>

#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <string.h>
#include "common.h"
/* #include "boost.h" */

#define BUFFER_SIZE 1000000
//#define DEBUG



double res[BUFFER_SIZE];
double temp1[BUFFER_SIZE];
double temp2[BUFFER_SIZE];

//matrix operation
//x:the tensor 3d
//y:the filter 4d
//z:the stride
CAMLprim value 
c_deri_tensor (value x,value y,value z){

#ifdef DEBUG
	printf("now in the c_deri_tensor function\n");
#endif
	
	//obtain the data from value type
	double* x_val = Caml_ba_data_val(x);
	double* y_val = Caml_ba_data_val(y);

	//obtain the properties pointer from the value type
	struct caml_ba_array *x_pro =Caml_ba_array_val(x);
	struct caml_ba_array *y_pro =Caml_ba_array_val(y);
	
	//check the shape of the two matrixs
	int x_dimension = x_pro -> num_dims;
	int y_dimension = y_pro -> num_dims;

	if( x_dimension != 3 || y_dimension != 4){
		printf("c_deri_tensor:dimension error \n");
		printf("left operand dimension number: %d,the right dimension number: %d\n",x_dimension,y_dimension);
		exit(DIMENSION_NOT_QUALIFIED);
	}

#ifdef DEBUG
	printf("first judgement\n");
#endif
	//define the shape of the two matrix
	int x_d,x_r,x_c;
	int y_f,y_d,y_r,y_c;

	//obtain the depth row and col of x
	x_d = x_pro -> dim[0];
	x_r = x_pro -> dim[1];
	x_c = x_pro -> dim[2];

	//obtain the filter depth row and col of y
	y_f = y_pro -> dim[0];
	y_d = y_pro -> dim[1];
	y_r = y_pro -> dim[2];
	y_c = y_pro -> dim[3];

	//check if the depth matchs
	if( x_d != y_f ){
		printf("c_deri_tensor:depth not match\n");
		printf("the depth of left operand %d\n",x_d);
		printf("the depth of right operand %d\n",y_f);
		exit(SHAPE_NOT_MATCH);
	}	


/* #ifdef DEBUG */
	//traverse the x
	/* for ( int i = 0; i<x_r ; i++){ */
	/* 	for (int j = 0;j<x_c;j++) */
	/* 		printf(" %f",x_val[i*x_c+j]); */
	/* 	printf ("\n"); */
	/* } */

	//traverse the y
	/* for ( int i = 0; i<y_r ; i++){ */
	/* 	for (int j = 0;j<y_c;j++) */
	/* 		printf(" %f",y_val[i*y_c+j]); */
	/* 	printf ("\n"); */
	/* } */
/* #endif */

	//prepare the ca: insert zero and pad
	//put the result in the temp1 buffer
	int stride = Int_val(z);
	int ins_d  = x_d;
	int ins_r;
	int ins_c;

	//insert zeros accocrding to the stride
	//the final result is put into the temp2
	if (stride > 1){
		int zeros = stride - 1;

		int z_x_d = x_d;
		int z_x_r = x_r + (x_r - 1) * zeros;
		int z_x_c = x_c + (x_c - 1) * zeros;



		for(int i = 0; i < z_x_d; i ++)
			for(int j = 0; j < z_x_r; j++)
				for(int k = 0; k < z_x_c; k++){
					if(j % stride != 0 || k % stride != 0)
						//shape of temp1: z_x_d z_x_r z_x_c
						//temp1[i][j][k] = 0.
						temp1[i*z_x_r*z_x_c + j*z_x_c + k] = 0.;
					else
						//shape of x_val: x_d x_r x_c
						//temp1[i][j][k] = x_val[i][j/stride][k/stride];
						temp1[i*z_x_r*z_x_c + j*z_x_c + k] = x_val[i*x_r*x_c + (j/stride)*x_c + (k/stride)];
				}

		memcpy(temp2,temp1,sizeof(double)*z_x_d*z_x_r*z_x_c);
		ins_r = z_x_r;
		ins_c = z_x_c;
	}
	else{
		memcpy(temp2,x_val,sizeof(double)*x_d*x_r*x_c);
		ins_r = x_r;
		ins_c = x_c;
	}

	//pad zeros according to the 
	//TRICK: the programmer needs to assure that the convlution kernel is square y_r = y_c
	int pd = y_r - 1;
	int pad_d = ins_d;
	int pad_r = ins_r + 2 * pd;
	int pad_c = ins_c + 2 * pd;

	//prepare for the result buffer
	if(pad_d * pad_r * pad_c > BUFFER_SIZE - 1) {
		printf ("c_deri_tensor:buffer is not enough");
		exit(BUFFER_OVERFLOW);
	}

	for(int i = 0; i < pad_d; i++)
		for(int j = 0; j < pad_r; j++)
			for(int k = 0; k < pad_c; k++){
				if( j >= pd && j < pad_r - pd && k >= pd && k < pad_c - pd)
					/* temp1[i][j][k] = temp2[i][j-pd][k-pd]; */
					temp1[i*pad_r*pad_c + j * pad_c + k] = temp2[i*ins_r*ins_c + (j-pd)*ins_c + (k-pd)];
				else 
					temp1[i*pad_r*pad_c + j * pad_c + k] = 0.;
			}	

#ifdef DEBUG
	printf("c_deri_tensor[DEBUG MODE]:the insert and pad result");
	//traverse the pad result
	for(int i = 0; i < pad_d; i++,printf("\n"))
		for(int j = 0; j < pad_r; j++,printf("\n"))
			for(int k = 0; k < pad_c; k++)
				printf("%.0lf ",temp1[i*pad_r*pad_c + j*pad_c + k]);
	
	
#endif
	//traditionally the filter needs to be reversed before calculating the result buffer
	//but, we do not, in sake of performance
	

	int res_d = y_d;
	int res_r = pad_r - y_r + 1;
	int res_c = pad_c - y_c + 1;

	//init the matrix c
	memset(res,0,sizeof(double) * res_r * res_c * res_d);
	
	//calculate the res value
	for(int i = 0; i < res_d; i++)
		for(int j = 0; j < res_r; j++)
			for(int k = 0; k < res_c; k++)
				for (int f = 0; f < y_f; f++)
					for (int r = 0; r < y_r; r++)
						for (int c = 0; c < y_c; c++){
						      /* res[i][j][k]+=(y_val[][j*stride+r][k*stride+c]*y_val[i][d][r][c]) */
						      int res_sub = i * res_r * res_c + j * res_c + k;
						      int x_sub = f * pad_r * pad_c + (j + r) * pad_c + (k + c);
						      int y_sub = f * y_d * y_r * y_c + i * y_r * y_c + (y_r - r - 1) * y_c + (y_c - c - 1);
						      res[res_sub] += (temp1[x_sub] * y_val[y_sub]);
						}
	
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc_dims(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,3,res,res_d,res_r,res_c);
}

//matrix operation
//x:the tensor 3d
//y:the filter 4d
//z:the stride
CAMLprim value 
c_deri_filter (value x,value y,value z){

#ifdef DEBUG
	printf("now in the c_deri_filter function\n");
#endif
	
	//obtain the data from value type
	double* x_val = Caml_ba_data_val(x);
	double* y_val = Caml_ba_data_val(y);

	//obtain the properties pointer from the value type
	struct caml_ba_array *x_pro =Caml_ba_array_val(x);
	struct caml_ba_array *y_pro =Caml_ba_array_val(y);
	
	//check the shape of the two matrixs
	int x_dimension = x_pro -> num_dims;
	int y_dimension = y_pro -> num_dims;

	if( x_dimension != 3 || y_dimension != 3){
		printf("c_deri_filter:dimension error \n");
		printf("left operand dimension number: %d,the right dimension number: %d\n",x_dimension,y_dimension);
		exit(DIMENSION_NOT_QUALIFIED);
	}

#ifdef DEBUG
	printf("first judgement\n");
#endif
	//define the shape of the two matrix
	int x_d,x_r,x_c;
	int y_d,y_r,y_c;

	//obtain the depth row and col of x
	x_d = x_pro -> dim[0];
	x_r = x_pro -> dim[1];
	x_c = x_pro -> dim[2];

	//obtain the filter depth row and col of y
	y_d = y_pro -> dim[0];
	y_r = y_pro -> dim[1];
	y_c = y_pro -> dim[2];



/* #ifdef DEBUG */
	//traverse the x
	/* for ( int i = 0; i<x_r ; i++){ */
	/* 	for (int j = 0;j<x_c;j++) */
	/* 		printf(" %f",x_val[i*x_c+j]); */
	/* 	printf ("\n"); */
	/* } */

	//traverse the y
	/* for ( int i = 0; i<y_r ; i++){ */
	/* 	for (int j = 0;j<y_c;j++) */
	/* 		printf(" %f",y_val[i*y_c+j]); */
	/* 	printf ("\n"); */
	/* } */
/* #endif */

	//prepare the ca: insert zero and pad
	//put the result in the temp1 buffer
	int stride = Int_val(z);
	int ins_d  = x_d;
	int ins_r;
	int ins_c;

	//insert zeros accocrding to the stride
	//the final result is put into the temp2
	if (stride > 1){
		int zeros = stride - 1;

		int z_x_d = x_d;
		int z_x_r = x_r + (x_r - 1) * zeros;
		int z_x_c = x_c + (x_c - 1) * zeros;



		for(int i = 0; i < z_x_d; i ++)
			for(int j = 0; j < z_x_r; j++)
				for(int k = 0; k < z_x_c; k++){
					if(j % stride != 0 || k % stride != 0)
						//shape of temp1: z_x_d z_x_r z_x_c
						//temp1[i][j][k] = 0.
						temp1[i*z_x_r*z_x_c + j*z_x_c + k] = 0.;
					else
						//shape of x_val: x_d x_r x_c
						//temp1[i][j][k] = x_val[i][j/stride][k/stride];
						temp1[i*z_x_r*z_x_c + j*z_x_c + k] = x_val[i*x_r*x_c + (j/stride)*x_c + (k/stride)];
				}

		memcpy(temp2,temp1,sizeof(double)*z_x_d*z_x_r*z_x_c);
		ins_r = z_x_r;
		ins_c = z_x_c;
	}
	else{
		memcpy(temp2,x_val,sizeof(double)*x_d*x_r*x_c);
		ins_r = x_r;
		ins_c = x_c;
	}

#ifdef DEBUG
	printf("c_deri_filter[DEBUG MODE]:the insert and pad result\n");
	//traverse the pad result
	for(int i = 0; i < ins_d; i++,printf("\n"))
		for(int j = 0; j < ins_r; j++,printf("\n"))
			for(int k = 0; k < ins_c; k++)
				printf("%.0lf ",temp2[i*ins_r*ins_c + j*ins_c + k]);
	
	printf("c_deri_filter[DEBUG MODE]:the tensor info\n");
	//traverse the pad result
	for(int i = 0; i < y_d; i++,printf("\n"))
		for(int j = 0; j < y_r; j++,printf("\n"))
			for(int k = 0; k < y_c; k++)
				printf("%.0lf ",y_val[i*y_r*y_c + j*y_c + k]);
	
#endif
	
	int res_f = x_d;
	int res_d = y_d;
	int res_r = y_r - ins_r + 1;
	int res_c = y_c - ins_c + 1;

	//init the matrix c
	memset(res,0,sizeof(double) * res_f * res_r * res_c * res_d);
	
	//calculate the res value
	for(int f = 0; f < res_f; f++)
		for(int i = 0; i < res_d; i++)
			for(int j = 0; j < res_r; j++)
				for(int k = 0; k < res_c; k++)
					for (int r = 0; r < ins_r; r++)
						for (int c = 0; c < ins_c; c++){
						      /* res[f][i][j][k]+=temp2[f][r][c]*y_val[i][j+r][k+c]*/
						      int res_sub = f * res_d * res_r * res_c + i * res_r * res_c + j * res_c + k;
						      int x_sub = f * ins_r * ins_c + r * ins_c + c;
						      int y_sub = i * y_r * y_c + (j+r) * y_c + (k+c);
						      res[res_sub] += (temp2[x_sub] * y_val[y_sub]);
						}
	
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc_dims(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,4,res,res_f,res_d,res_r,res_c);
}
//matrix operation
//x:the ca
//y:the filter
//z:the stride
CAMLprim value 
c_layer_conv3d (value x,value y,value z){

#ifdef DEBUG
	printf("now in the c_layer_conv3d function\n");
#endif
	
	//obtain the data from value type
	double* x_val = Caml_ba_data_val(x);
	double* y_val = Caml_ba_data_val(y);

	//obtain the properties pointer from the value type
	struct caml_ba_array *x_pro =Caml_ba_array_val(x);
	struct caml_ba_array *y_pro =Caml_ba_array_val(y);
	
	//check the shape of the two matrixs
	int x_dimension = x_pro -> num_dims;
	int y_dimension = y_pro -> num_dims;

	if( x_dimension != 3 || y_dimension != 4){
		printf("c_layer_conv_ed:dimension error \n");
		printf("left operand dimension number: %d,the right dimension number: %d\n",x_dimension,y_dimension);
		exit(DIMENSION_NOT_QUALIFIED);
	}

#ifdef DEBUG
	printf("first judgement\n");
#endif
	//define the shape of the two matrix
	int x_d,x_r,x_c;
	int y_f,y_d,y_r,y_c;

	//obtain the depth row and col of x
	x_d = x_pro -> dim[0];
	x_r = x_pro -> dim[1];
	x_c = x_pro -> dim[2];

	//obtain the filter depth row and col of y
	y_f = y_pro -> dim[0];
	y_d = y_pro -> dim[1];
	y_r = y_pro -> dim[2];
	y_c = y_pro -> dim[3];

	//check if the depth matchs
	if( x_d != y_d ){
		printf("c_layer_conv3d:the deri result and filter not match\n");
		printf("the depth of left operand %d\n",x_d);
		printf("the depth of right operand %d\n",y_f);
		exit(SHAPE_NOT_MATCH);
	}	


/* #ifdef DEBUG */
	//traverse the x
	/* for ( int i = 0; i<x_r ; i++){ */
	/* 	for (int j = 0;j<x_c;j++) */
	/* 		printf(" %f",x_val[i*x_c+j]); */
	/* 	printf ("\n"); */
	/* } */

	//traverse the y
	/* for ( int i = 0; i<y_r ; i++){ */
	/* 	for (int j = 0;j<y_c;j++) */
	/* 		printf(" %f",y_val[i*y_c+j]); */
	/* 	printf ("\n"); */
	/* } */
/* #endif */

	//get the stride
	int stride = Int_val(z);

	//calculate the result shape
	long  res_d = y_f;
	long  res_r = (x_r - y_r)/stride + 1;
	long  res_c = (x_c - y_c)/stride + 1;

	//prepare for the result buffer
	if(res_d * res_r * res_c > BUFFER_SIZE - 1) {
		printf ("c_layer_conv3d:buffer is not enough");
		exit(BUFFER_OVERFLOW);
	}
	
	//init the matrix c
	memset(res,0,sizeof(double) * res_r * res_c * res_d);
	
	//calculate the res value
	for(int i = 0; i < res_d; i++)
		for(int j = 0; j < res_r; j++)
			for(int k = 0; k < res_c; k++)
				for (int d = 0; d < x_d; d++)
					for(int r = 0; r < y_r; r++)
						for (int c = 0; c < y_c; c++){
							/* res[i][j][k]+=(x_val[d][j*stride+r][k*stride+c]*y_val[i][d][r][c]) */
							int res_sub = i * res_r * res_c + j * res_c + k;
							int x_sub = d * x_r * x_c + (j * stride + r) * x_c + (k * stride + c);
							int y_sub = i * y_d * y_r * y_c + d * y_r * y_c + r * y_c + c;
							res[res_sub] += (x_val[x_sub] * y_val[y_sub]);
						}
	


/* #ifdef DEBUG */
/* 	//traverse the z */
/* 	for ( int i = 0; i<res_r ; i++){ */
/* 		for (int j = 0;j<res_c;j++) */
/* 			printf(" %f",y_val[i*res_c+j]); */
/* 		printf ("\n"); */
/* 	} */
/* #endif */

	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc_dims(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,3,res,res_d,res_r,res_c);
}
