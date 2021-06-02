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
//matrix operation
CAMLprim value 
cuda_mat_mul (value x,value y);
