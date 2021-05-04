
int numel_by_shape (int shape [],int len){
	int res = 1;
	for (int  i = 0; i< len ; i++){
		res *= shape[i];
	}
	return res;
}
