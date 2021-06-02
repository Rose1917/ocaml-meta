dylib:
	nvcc --shared -o libcuda.so mat_cuda.cu --compiler-options '-L /usr/local/cuda/lib -lcudart -fPIC' --ptxas-options=-v
static:
	nvcc -o libcuda.a mat_cuda.cu -lib 
