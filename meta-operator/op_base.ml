open Owl_dense_ndarray_d

(*defination of the old fasion reindex *)
let reindex_base input output_shape ~map_func = 
  let f = function input_index -> get input (map_func input_index) in
  init_nd output_shape f

(*defination of the old fasion reindex-reduce *)
let reindex_reduce_base input output_shape ~map_func = 
    let get_aux_tensor i = 
        let aux_fun aux_index e = let new_index = map_func aux_index in 
            if Stdlib.(=) new_index  i then e else 0.   in
        mapi_nd aux_fun input in
    let f = function input_index -> sum' (get_aux_tensor input_index) in
init_nd output_shape f

(*defination of unary element-wise meta-operator*)
let element_wise_unary_base input ~map_func =
    let output_shape = shape input in
    let f = function index -> map_func (get input index) in
init_nd output_shape f;;

(*defination of binary element-wise meta-operator*)
let element_wise_binary_base input_1 input_2 ~map_func = 
    let output_shape = shape input_1 in
    let f = function index -> map_func (get input_1 index) (get input_2 index) in
init_nd output_shape f;;

(*defination of ternary element-wise meta-operator*)
let element_wise_ternary_base input_1 input_2 input_3 ~map_func = 
    let output_shape = shape input_1 in
    let f = function index -> map_func (get input_1 index) (get input_2 index) (get input_3 index) in
init_nd output_shape f;;
