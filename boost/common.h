#ifndef MAIN_H
#define MAIN_H
/*error variant*/
typedef enum ERROR_TYPE{
	DIMENSION_NOT_QUALIFIED = 1,
	SHAPE_NOT_MATCH,
	MEMORY_ALLOCATION_FAILED,
	BOOST_TYPE_NOT_SPECIFIED,
	BUFFER_OVERFLOW
}ERROR_TYPE;

/*boost type*/
typedef enum BOOST_TYPE{
	AVX_BOOST = 0,
	FMA_BOOST = 1,
	OMP_BOOST = 2,
}BOOST_TYPE;


#endif
