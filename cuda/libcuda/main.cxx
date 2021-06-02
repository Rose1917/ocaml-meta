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

CAMLprim value 
cuda_mat_mul (value x,value y);

int main(){
	cuda_mat_mul (NULL,NULL);
	return 0;
}
