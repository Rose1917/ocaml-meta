#include <stdio.h>
#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/callback.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <string.h>
#include "common.h"
#include "boost.h"
#include "util.h"
#define DEBUG

#define VAL_INT_ARR(array,i) Int_val(Field(array,i))
#define LEN_ARR(array) Wosize_val(array)




//matrix operation
CAMLprim value 
c_reindex (value src_t, value out_shape){
	
	//avoid the ocaml grabage to collect the shape value
	CAMLparam1(out_shape);

	//obtain the data from value type
	double* src_val = Caml_ba_data_val(src_t);

	//calculate the result element
	int len = LEN_ARR(out_shape);
	int res_numele = 1;

        //the result dims,since the maxmium dimension is 16.
	long dims [18];
	for ( int i = 0 ; i < len ; i++){
		dims[i]=VAL_INT_ARR(out_shape,i);
		res_numele *= dims[i];
	}

	//prepare for the result buffer
	double *res = (double *) malloc( sizeof (double *) * res_numele); 
	if (res == NULL){
		printf ("c_reindex:can not access the memory result need");
		exit (MEMORY_ALLOCATION_FAILED);
	}
	
	//init the result buffer
	memset(res,0,sizeof(double) * res_numele);
	
	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("reindex_map_func");

	//set the res buffer
	for (int i = 0 ; i < res_numele ; i++){
		res[i] = src_val[Int_val(caml_callback(*map_func,Val_int(i)))];
		printf("iteration %d:%d -> %d\n",i,i,Int_val(caml_callback(*map_func,Val_int(i))));
        }	

	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,len,res,dims);
}

CAMLprim value 
c_reindex_reduce (value src_t, value out_shape){
	
	//avoid the ocaml grabage to collect the shape value
	CAMLparam1(out_shape);

	//obtain the data and property pointer from value type
	double* src_val = Caml_ba_data_val(src_t);
	struct caml_ba_array* src_pro = Caml_ba_array_val(src_t);
	
	//calculate the src element_number and dimension
	int src_dim = src_pro -> num_dims;
	int src_numele = 1;
	for ( int i = 0; i < src_dim ; i++)
		src_numele *= src_pro -> dim[i];

	//calculate the result element
	int len = LEN_ARR(out_shape);
	int res_numele = 1;

        //the result dims:the maxminum dimension is 16
	long dims [18];
	for ( int i = 0 ; i < len ; i++){
		dims[i]=VAL_INT_ARR(out_shape,i);
		res_numele *= dims[i];
	}

	//prepare for the result buffer
	double *res = (double *) malloc( sizeof (double *) * res_numele); 
	if (res == NULL){
		printf ("c_reindex:can not access the memory result need");
		exit (MEMORY_ALLOCATION_FAILED);
	}
	
	//init the result buffer
	memset(res,0,sizeof(double) * res_numele);
	
	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("reindex_reduce_map_func");

	//set the res buffer
	for (int i = 0 ; i < res_numele ; i++)
		for (int j = 0; j < src_numele ; j++)
			if ( i == Int_val(caml_callback(*map_func,Val_int(j))))
			res[i] += src_val[j];
	
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,len,res,dims);
}
