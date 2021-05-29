#include <stdio.h>
#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <string.h>
#include "common.h"
#include "boost.h"

#define BUFFER_SIZE 1000000
//#define DEBUG



double res[BUFFER_SIZE];

//matrix operation
CAMLprim value 
c_mat_mul (value x,value y, value z){

#ifdef DEBUG
	printf("now in the c_mat_mul function\n");
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

	if( x_dimension != 2 || y_dimension != 2){
		printf("c_mat_mul:dimension error \n");
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
		printf("c_mat_mul:shape not match\n");
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
		printf ("c mat_mul:buffer is not enough");
		exit(BUFFER_OVERFLOW);
	}
	
	//init the matrix c
	memset(res,0,sizeof(double) * res_r * res_c);
	

	//obtain the boost type from the argument
	int boost_type = Int_val(z);

	//call the boost function
	switch ( boost_type ){
		case AVX_BOOST:
			#ifdef DEBUG
			printf("BOOST TYPE:AVX\n");
			#endif
			mul_avx(x_val,y_val,res,x_r,y_c,x_c);	
			break;
		case FMA_BOOST:
			#ifdef DEBUG
			printf("BOOST TYPE:FMA\n");
			#endif
			mul_fma(x_val,y_val,res,x_r,y_c,x_c);	
			break;
		case OMP_BOOST:
			#ifdef DEBUG
			printf("BOOST TYPE:OMP\n");
			#endif
			mul_omp(x_val,y_val,res,x_r,y_c,x_c);	
			break;
		default:
			printf("c_mat_mul:the boost type is not recognized.\n");
			printf("exit now.\n");
			exit(BOOST_TYPE_NOT_SPECIFIED);
	}
		
#ifdef DEBUG
	//traverse the z
	for ( int i = 0; i<res_r ; i++){
		for (int j = 0;j<res_c;j++)
			printf(" %f",y_val[i*res_c+j]);
		printf ("\n");
	}
#endif


	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc_dims(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,2,res,res_r,res_c);
}
