#define CAML_NAME_SPACE
#include <stdio.h>
#include <stdlib.h>
#include <caml/bigarray.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/callback.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/custom.h>
#include <caml/threads.h>
#include <caml/intext.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>
#include <string.h>
#include "common.h"
#include "boost.h"
#include "util.h"
#define DEBUG
#define BUFFER_SIZE 100000
#define VAL_INT_ARR(array,i) Int_val(Field(array,i))
#define LEN_ARR(array) Wosize_val(array)


static int flag_num = 0;
static void print_flat (){
	printf("flag:%d\n",flag_num++);
}


double res [BUFFER_SIZE];
//matrix operation
CAMLprim value 
c_reindex (value src_t, value out_shape){
	
	//avoid the ocaml grabage to collect the shape value
	//CAMLparam1(out_shape);

	//printf("c_reindex: c boost reindex\n");
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

	

	
	//init the result buffer
	//memset(res,0,sizeof(res));
	

	
	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("reindex_map_func");

	//set the res buffer
	for (int i = 0 ; i < res_numele ; i++){
		int src_index = Int_val(caml_callback(*map_func,Val_int(i)));
		res[i] = src_val[src_index];
        }	
	

	
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,len,res,dims);
}

CAMLprim value 
c_reindex_reduce (value src_t, value out_shape){
	
	//avoid the ocaml grabage to collect the shape value
	//CAMLparam1(out_shape);

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

	
	//init the result buffer
	memset(res,0,sizeof(double)*res_numele);
	
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

CAMLprim value 
c_element_wise_unary (value src_t){
	
	//print_flat();
	CAMLparam0();
	//obtain the data and property pointer from value type
	
	double* src_val = Caml_ba_data_val(src_t);
	struct caml_ba_array* src_pro = Caml_ba_array_val(src_t);
	
	//calculate the src element_number and dimension
	int src_dim = src_pro -> num_dims;
	int src_numele = 1;
	long dims [18];
	for ( int i = 0; i < src_dim ; i++){
		src_numele *= src_pro -> dim[i];
		dims[i] = src_pro -> dim[i];
	}

	int res_numele = src_numele;

	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("element_wise_unary_map_func");

	//set the res buffer
	//memset(res,0,sizeof(res));
	
	for (int i = 0 ; i < res_numele ; i++){
		double tmp = Double_val(caml_callback(*map_func,caml_copy_double(src_val[i])));
		res[i]     = tmp;
	}
        //print_flat();	
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,src_dim,res,dims);
}
CAMLprim value 
c_element_wise_binary (value src_t_1, value src_t_2){
	
	CAMLparam0();
	//obtain the data and property pointer from value type
	
	double* src_val_1 = Caml_ba_data_val(src_t_1);
	double* src_val_2 = Caml_ba_data_val(src_t_2);
	struct caml_ba_array* src_pro_1 = Caml_ba_array_val(src_t_1);
	struct caml_ba_array* src_pro_2 = Caml_ba_array_val(src_t_2);

	//print_flat();
	
	//calculate the src element_number and dimension
	int src_dim_1 = src_pro_1 -> num_dims;
	int src_dim_2 = src_pro_2 -> num_dims;

	if( src_dim_1 != src_dim_2){
		printf("element_wise_binary:the dims of the two tensor are not the same\n");
		printf("the former one is %d, and the latter one is %d",src_dim_1,src_dim_2);
		exit(SHAPE_NOT_MATCH);
	}

	//check the dim iterally and calculate the number by the way
        int res_numele = 1;	
	long dims [18];
	for (int i = 0; i < src_dim_1 ; i++){
		if ( src_pro_1 -> dim [i] != src_pro_2 -> dim [i]){
			printf ("c_element_wise_binary:the shape of the two tensor are not the same\n");
			printf ("dim %d: the former is %d, the latter is %d\n",i,src_pro_1 -> dim[i],src_pro_2 -> dim [i]);
			exit(SHAPE_NOT_MATCH);
		}
		res_numele *= src_pro_1 -> dim[i];
		dims[i] = src_pro_1 -> dim[i];
	}
	
	
	
	//print_flat();

	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("element_wise_binary_map_func");

	//set the res buffer
	for (int i = 0 ; i < res_numele ; i++){
		double tmp = Double_val(caml_callback2(*map_func,caml_copy_double(src_val_1[i]),caml_copy_double(src_val_2[i])));
		res[i]     = tmp;
	}
	//print_flat();

	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,src_dim_1,res,dims);
}
CAMLprim value 
c_element_wise_ternary (value src_t_1, value src_t_2, value src_t_3){
	
	CAMLparam0();
	//obtain the data and property pointer from value type
	
	double* src_val_1 = Caml_ba_data_val(src_t_1);
	double* src_val_2 = Caml_ba_data_val(src_t_2);
	double* src_val_3 = Caml_ba_data_val(src_t_3);
	struct caml_ba_array* src_pro_1 = Caml_ba_array_val(src_t_1);
	struct caml_ba_array* src_pro_2 = Caml_ba_array_val(src_t_2);
	struct caml_ba_array* src_pro_3 = Caml_ba_array_val(src_t_3);
	
	//calculate the src element_number and dimension
	int src_dim_1 = src_pro_1 -> num_dims;
	int src_dim_2 = src_pro_2 -> num_dims;
	int src_dim_3 = src_pro_3 -> num_dims;

	if( src_dim_1 != src_dim_2 || src_dim_2 != src_dim_3){
		printf("element_wise_ternary:the dims of the three tensor are not the same\n");
		printf("the first one is %d, and the second one is %d,the third one is %d\n",src_dim_1,src_dim_2,src_dim_3);
		exit(SHAPE_NOT_MATCH);
	}

	//check the dim iterally and calculate the number by the way
        int res_numele = 1;	
	long dims [18];
	for (int i = 0; i < src_dim_1 ; i++){
		if ( src_pro_1 -> dim [i] != src_pro_2 -> dim [i] ||src_pro_2 -> dim [i] != src_pro_3 -> dim [i]){
			printf ("c_element_wise_ternary:the shape of the three tensor are not the same\n");
			printf ("dim %ld: the first is %ld, the second is %ld, the third one is %ld\n",i,src_pro_1 -> dim[i],src_pro_2 -> dim [i],src_pro_3 -> dim[i]);
			exit(SHAPE_NOT_MATCH);
		}
		res_numele *= src_pro_1 -> dim[i];
		dims[i] = src_pro_1 -> dim[i];
	}
	
	

        //memset(res,0,sizeof(res));

	//cache the map function
	static value* map_func = NULL;
	if(map_func == NULL)
		map_func = caml_named_value("element_wise_ternary_map_func");

	//set the res buffer
	for (int i = 0 ; i < res_numele ; i++){
		double tmp = Double_val(caml_callback3(*map_func,caml_copy_double(src_val_1[i]),caml_copy_double(src_val_2[i]),caml_copy_double(src_val_3[i])));
		res[i]     = tmp;
	}
	//call the function and if it works properly, we need to pack it to a bigarray so we can send it back
	return caml_ba_alloc(CAML_BA_FLOAT64|CAML_BA_C_LAYOUT,src_dim_1,res,dims);
}
