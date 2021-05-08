include Var.D
open Bigarray

exception Index_out_of_bound of string
exception Shape_error of string

(* the creation operator*)
 

(*this file implement the basic ndarray and the function of it which is like numpy*)

(*define the basic type*)
type base_t = (float,float64_elt,c_layout)Bigarray.Genarray.t

let shape (x:base_t) = 
  Genarray.dims x 

let num_dims x = 
  Genarray.num_dims x
 
let get (x:base_t) index = 
  Genarray.get x index

let init_nd output_shape f = 
  Genarray.init float64 c_layout output_shape f

let numel_of_shape shape =
  let f x y = x * y in
  Array.fold_left f 1 shape

let numel x = 
  let arr = shape x in
  numel_of_shape arr 

let flatten x = 
  reshape x [|(numel x);|]

let copy x = 
  let s = shape x in
  let f i = get x i in
  init_nd s f

let reverse x = 
  let l = Array.to_list x in
  let ll = List.rev l in
  Array.of_list ll


let cal_stride shape = 
  let l = Array.length shape in
  let s = Array.copy shape in
  let tmp = ref 1 in
  for i = l - 1 downto 0 do
    s.(i) <- !tmp;
    tmp := !tmp * shape.(i);
  done;
  s

let index_1d_nd one_d shape = 
  let sd = cal_stride shape in
  let res = Array.copy shape in
  let l = Array.length shape in
  let temp = ref one_d in
  for i = 0 to l - 1 do
    res.(i) <- (!temp / sd.(i)); 
    temp := !temp mod sd.(i);
  done;
  res

let index_nd_1d nd shape = 
  let sd = cal_stride shape in
  let res = ref 0 in
  let l = Array.length shape in
  for i = 0 to l - 1 do
    res := !res + (sd.(i) * nd.(i))
  done;
  !res

let fold_left f left_val t = 
  let res = ref left_val in
  let length = numel t in
  let flat_t = flatten t |> array1_of_genarray in
  for i = 0 to length - 1 do
    res := f !res (Array1.unsafe_get flat_t i)
  done;
  !res

let mapi_nd f s = 
  let fi index = 
          let ele = get s index in
          f index ele  in
  init_nd (shape s) fi   

let sum_scalar s =
  let f = (+.) in
  fold_left f 0. s

let sequential ?(a=0.) ?(step=1.0) shape =
  let v = ref a in
  let n = numel_of_shape shape in
  let res_raw = Array1.create float64 c_layout n in
  for i = 0 to n - 1 do
    Array1.unsafe_set res_raw i !v;
    v := !v +. step;
  done;
  let res_vec = genarray_of_array1 res_raw in
  reshape res_vec shape

let random ?(bound = 1.) shape = 
  Random.self_init ();
  let n = numel_of_shape shape in
  let res_raw = Array1.create float64 c_layout n in
  for i = 1 to n - 1 do 
    Array1.unsafe_set res_raw i (Random.float bound); 
  done;
  let res_vec = genarray_of_array1 res_raw in
  reshape res_vec shape 

(* some creation functions *)
let zeros shape = 
  let res = Genarray.create float64 c_layout shape in
  Genarray.fill res 0.;
  res
let ones shape = 
  let res = Genarray.create float64 c_layout shape in
  Genarray.fill res 1.;
  res

let ns  x shape = 
  let res = Genarray.create float64 c_layout shape in
  Genarray.fill res x;
  res

let zeros_like tensor = 
  let s = shape tensor in
  zeros s

let ones_like tensor = 
  let s = shape tensor in
  ones s

let ns_like x tensor = 
  let s = shape tensor in
  ns x s

external c_reindex : base_t -> int array -> base_t = "c_reindex" 
(*defination of the old fasion reindex *)
let reindex_boost input output_shape ~map_func = 
  let map_func' index_i = 
    let index_nd = index_1d_nd index_i output_shape in
    let nd_res = map_func index_nd in
    index_nd_1d nd_res (shape input)
  in
  Callback.register "reindex_map_func" map_func';
  let res = c_reindex input output_shape in
  copy res


external c_reindex_reduce : base_t -> int array -> base_t = "c_reindex_reduce" 
(*defination of the old fasion reindex-reduce *)
let reindex_reduce_boost input output_shape ~map_func = 
  let map_func' index_i = 
    let index_nd = index_1d_nd index_i (shape input) in
    let nd_res = map_func index_nd in
    index_nd_1d nd_res output_shape
  in
  Callback.register "reindex_reduce_map_func" map_func';
  let res = c_reindex_reduce input output_shape in
  copy res


external c_element_wise_unary : base_t -> base_t = "c_element_wise_unary"
(*defination of unary element-wise meta-operator*)
let element_wise_unary_boost input ~map_func =
  Callback.register "element_wise_unary_map_func" map_func;
  let res = c_element_wise_unary input in
  copy res

external c_element_wise_binary : base_t -> base_t -> base_t = "c_element_wise_binary"
(*defination of binary element-wise meta-operator*)
let element_wise_binary_boost input_1 input_2 ~map_func = 
  Callback.register "element_wise_binary_map_func" map_func;
  let res = c_element_wise_binary input_1 input_2 in
  copy res

external c_element_wise_ternary : base_t -> base_t -> base_t -> base_t = "c_element_wise_ternary"
(*defination of ternary element-wise meta-operator*)
let element_wise_ternary_boost input_1 input_2 input_3 ~map_func = 
  Callback.register "element_wise_ternary_map_func" map_func;
  let res = c_element_wise_ternary input_1 input_2 input_3 in
  copy res

(*defination of the old fasion reindex *)
let reindex_caml input output_shape ~map_func = 
  let f = function input_index -> get input (map_func input_index) in
  init_nd output_shape f

(*defination of the old fasion reindex-reduce *)
let reindex_reduce_caml input output_shape ~map_func = 
    let get_aux_tensor i = 
        let aux_fun aux_index e = let new_index = map_func aux_index in 
            if Stdlib.(=) new_index  i then e else 0.   in
        mapi_nd aux_fun input in
    let f = function input_index -> sum_scalar (get_aux_tensor input_index) in
init_nd output_shape f

(*defination of unary element-wise meta-operator*)
let element_wise_unary_caml input ~map_func =
    let output_shape = shape input in
    let f = function index -> map_func (get input index) in
init_nd output_shape f

(*defination of binary element-wise meta-operator*)
let element_wise_binary_caml input_1 input_2 ~map_func = 
    let output_shape = shape input_1 in
    let f = function index -> map_func (get input_1 index) (get input_2 index) in
init_nd output_shape f

(*defination of ternary element-wise meta-operator*)
let element_wise_ternary_caml input_1 input_2 input_3 ~map_func = 
    let output_shape = shape input_1 in
    let f = function index -> map_func (get input_1 index) (get input_2 index) (get input_3 index) in
init_nd output_shape f

let reindex ?(bt=CAML)= 
  match (bt) with
  |CBOOST -> reindex_boost
  |CAML -> reindex_caml
;;

let reindex_reduce ?(bt=CAML)= 
  match (bt) with
  |CBOOST -> reindex_reduce_boost
  |CAML -> reindex_reduce_caml
;;

let element_wise_unary ?(bt=CAML)= 
  match (bt) with
  |CBOOST -> element_wise_unary_boost
  |CAML -> element_wise_unary_caml
;;

let element_wise_binary ?(bt=CAML)= 
  match (bt) with
  |CBOOST -> element_wise_binary_boost
  |CAML -> element_wise_binary_caml
;;

let element_wise_ternary ?(bt=CAML)= 
  match (bt) with
  |CBOOST -> element_wise_ternary_boost
  |CAML -> element_wise_ternary_caml
;;

(*general unary function*)
let abs ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.abs ~bt

let neg ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.neg ~bt

let floor ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.floor ~bt

let sqr ?(bt=CAML) = 
  let f x = x *. x in
  element_wise_unary ~map_func:f ~bt

let sqrt ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.sqrt ~bt

let log ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.log ~bt

let log2 ?(bt=CAML) = 
  let f x = (Float.log x) /. (Float.log 2.) in
  element_wise_unary ~map_func:f ~bt

let log10 ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.log10 ~bt

let exp ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.exp ~bt

let cos ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.cos ~bt

let sin ?(bt=CAML) = 
  element_wise_unary ~map_func:Float.sin ~bt

let tan ?(bt=CAML) = 
  let f x = (Float.sin x) /. (Float.cos x) in
  element_wise_unary ~map_func:f ~bt

let sum ?(bt=CAML) ~axis input = 
  let output_shape input_shape =
        let aux_f_1 i =
            let aux_i_f x = if (Stdlib.( >=) x axis) then Stdlib.(+) x  1 else x in
            input_shape.(aux_i_f i) in
    Array.init (Stdlib.(-) (Array.length input_shape) 1) aux_f_1 in
    let out_shape = output_shape (shape input) in

    let index_func input_index =
        let aux_f_2 i =
            let aux_i_n x = if (Stdlib.( >=) x axis) then Stdlib.(+) x  1 else x in
            input_index.(aux_i_n i) in
    Array.init ((Array.length input_index) - 1) aux_f_2 in
    reindex_reduce input out_shape ~map_func:index_func ~bt:bt;;

let sum' ?(bt=CAML) input = 
  let f _index = [|0|] in
  reindex_reduce input [|1|] ~map_func:f ~bt

let reci_procal ?(bt=CAML) = 
  let f x = 1. /. x in
  element_wise_unary ~map_func:f  ~bt

(*activation functions*)
let sigmoid   ?(bt=CAML)= 
  let f x = 
    let neg = Float.neg x in
    let exp = Float.exp neg in
    let plu = exp +. 1. in
    1. /. plu in
  element_wise_unary ~map_func:f ~bt

let relu   ?(bt=CAML)= 
  let f x = 
    if x > 0. then x else 0. in
  element_wise_unary ~map_func:f ~bt

let tanh   ?(bt=CAML)= 
  let f x = 
    let ex = Float.exp x in
    let e_neg_x = Float.exp (-.x) in
    (ex -. e_neg_x) /. (ex +. e_neg_x) in
  element_wise_unary ~map_func:f ~bt

let non_act ?(bt=CAML) =
  let f x =
    x in
  element_wise_unary ~map_func:f ~bt
(*general basic binary functions*)
let add  ?(bt=CAML)= 
  element_wise_binary ~map_func:Float.add ~bt:bt

let sub  ?(bt=CAML)= 
  element_wise_binary ~map_func:Float.sub ~bt:bt

let mul  ?(bt=CAML)= 
  element_wise_binary ~map_func:Float.mul ~bt:bt

let div ?(bt=CAML)= 
  element_wise_binary ~map_func:Float.div ~bt:bt

let add_scalar ?(bt=CAML) x y = 
  let f a = a +. y in
  element_wise_unary x ~map_func:f ~bt:bt

let sub_scalar ?(bt=CAML) x y = 
  let f a = a -. y in
  element_wise_unary x ~map_func:f ~bt:bt

let mul_scalar ?(bt=CAML) x y = 
  let f a = a *. y in
  element_wise_unary x ~map_func:f ~bt:bt

let div_scalar ?(bt=CAML) x y = 
  let f a = a /. y in
  element_wise_unary x ~map_func:f ~bt:bt


(* matrix operation *)
let transpose ?(bt=CAML) x = 
  let f index = [|index.(1);index.(0)|] in
  reindex x (reverse (shape x)) ~map_func:f ~bt


let broad_cast ?(bt=CAML) input target_shape oxis = 
  let f input_index =
    let aux_f i =
      let aux_i_f x = if (x >= oxis) then x + 1 else x in
    input_index.(aux_i_f i) in
   Array.init ((Array.length input_index) - 1) aux_f in
reindex input target_shape ~map_func:f ~bt:bt


let dot ?(bt=CAML) x y = 
  let shape_b = 
    let shape_x = shape x in
    let shape_y = shape y in
    [|shape_x.(0);shape_x.(1);shape_y.(1)|] in
  let x_bd = broad_cast x shape_b 2 in
  let y_bd = broad_cast y shape_b 0 in
  let raw_res = mul x_bd y_bd in
  sum ~axis:1 ~bt raw_res 



type boost_type = 
  |AVX_BOOST
  |FMA_BOOST
  |OMP_BOOST
  |DEFAULT
external c_mat_mul : base_t -> base_t -> boost_type -> base_t  = "c_mat_mul"

(*matrix operation*)
let boost = ref DEFAULT
let set_boost t =
  boost:= t

let mat_dot ?(bt=CAML) x y  = 
  match !boost with
  |DEFAULT -> dot x y ~bt
  |_ -> copy (c_mat_mul x y !boost)




(*misc *)
let print_vec x =
  print_endline "";
  let n = numel x in 
  for i = 0 to n - 1 do
    Printf.printf " %g" (get x [|i|]);
  done;
  print_endline ""

let print_mat x = 
  print_endline "";
  let s = shape x in
  let r = s.(0) in
  let c = s.(1) in
  for i = 0 to r - 1 do
    for j = 0 to c - 1 do
      Printf.printf " %g" (get x [|i;j|]);
    done;
    print_endline "";
  done

let print_cube x_t = 
  print_endline "";
  let s = shape x_t in
  let x = s.(0) in
  let y = s.(1) in
  let z = s.(2) in
  for i = 0 to x - 1 do
    for j = 0 to y - 1 do
      for k = 0 to z - 1 do
        Printf.printf " %g" (get x_t [|i;j;k|]);
      done;
      print_endline "";
    done;
    print_endline "";
  done

let print ?(prefix = "") x = 
  if (String.length prefix) = 0 then () else Printf.printf " %s" prefix;
  let dims = num_dims x in
  match dims with 
  |1 -> print_vec x 
  |2 -> print_mat x
  |3 -> print_cube x
  |_ -> raise (Shape_error "print:the shape is not supported")

let printf = Printf.printf 

