include Var.D
open Bigarray

exception Index_out_of_bound of string
exception Shape_error of string

(* the creation operator*)
 
(*this file implement the basic ndarray and the function of it which is like numpy*)

(*define the basic type*)
type base_t = (float,float64_elt,c_layout)Bigarray.Genarray.t

let shape x = 
  Genarray.dims x 

let num_dims x = 
  Genarray.num_dims x
 
let get x index = 
  Genarray.get x index

let init_nd output_shape f = 
  Genarray.init float64 c_layout output_shape f

let from32 x = 
  let f index =
    get x index in
  init_nd (shape x) f 

let numel_of_shape shape =
  let f x y = x * y in
  Array.fold_left f 1 shape
let reshape = 
  Bigarray.reshape
let numel x = 
  let arr = shape x in
  numel_of_shape arr 

let copy x = 
  let s = shape x in
  let f i = get x i in
  init_nd s f

let flatten x = 
  let x_copy = copy x in
  reshape x_copy [|(numel x);|]

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

let average_scalar s = 
  let n = s |> numel |> Float.of_int in
  let v = sum_scalar s in
  v /. n

let max_i x =
  let x = flatten x in
  let n = numel x in
  let max_index = ref (0) in
  let max_ele   = ref (Float.min_float) in
  for it = 0 to n - 1 do
    if ((Genarray.get x [|it|]) > !max_ele) then (max_ele := (Genarray.get x [|it|]);max_index := it)
  done;
  !max_index

let max x =
  let x = flatten x in
  let n = numel x in
  let max_index = ref (-1) in
  let max_ele   = ref (Float.min_float) in
  for it = 0 to n - 1 do
    if ((Genarray.get x [|it|]) > !max_ele) then (max_ele := (Genarray.get x [|it|]);max_index := it)
  done;
  !max_ele

let min_i x =
  let x = flatten x in
  let n = numel x in
  let min_index = ref (-1) in
  let min_ele   = ref (Float.max_float) in
  for it = 0 to n - 1 do
    if ((Genarray.get x [|it|]) < !min_ele) then (min_ele := (Genarray.get x [|it|]);min_index := it)
  done;
  !min_index


let min x =
  let x = flatten x in
  let n = numel x in
  let min_index = ref (-1) in
  let min_ele   = ref (Float.max_float) in
  for it = 0 to n - 1 do
    if ((Genarray.get x [|it|]) < !min_ele) then (min_ele := (Genarray.get x [|it|]);min_index := it)
  done;
  !min_ele


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

let compare_array x_d y_d = 
  let x_l = Array.length x_d in
  let y_l = Array.length y_d in
  if x_l != y_l then false else
        let y_cp = Array.copy y_d in 
        let f i e = if x_d.(i) == e then y_cp.(i) <- 1 else y_cp.(i) <- 0 in
        Array.iteri f y_cp;
        let res_int = Array.fold_left ( * ) 1 y_cp  in
        res_int == 1
let%test _ = compare_array [|2;3;4|] [|3;4;5|] = false
let%test _ = compare_array [|2;3;4;5|] [|2;3;4|] = false
let%test _ = compare_array [|2;3;4|] [|2;3;4|] = true
let%test _ = compare_array [|2;3;4|] [|2;3;4;5|] = false

let compare_shape x y = 
  let x_d = shape x in
  let y_d = shape y in
  compare_array x_d y_d

let%test _ = compare_shape (sequential [|3;4|]) (sequential [|3;4;5|]) = false
let%test _ = compare_shape (sequential [|3;4|]) (sequential [|3;4|]) = true
let%test _ = compare_shape (random [|3;4|]) (sequential [|3;4|]) = true

let compare x y = 
  let shape_cmp_res = compare_shape x y in
  if shape_cmp_res = false then false else 
    let x_flat = flatten x in
    let y_flat = flatten y in
    let n = numel x in
    let res = ref true in
    for i = 0 to n - 1 do
      let x_ele = get x_flat [|i;|] in
      let y_ele = get y_flat [|i;|] in
      res := (!res) && (x_ele = y_ele);
    done;
    !res

let%test _ = compare (sequential [|3;4|]) (sequential [|3;4;5|]) = false
let%test _ = compare (sequential [|3;4|]) (sequential [|3;4|]) = true
let%test _ = compare (random [|3;4|]) (sequential [|3;4|]) = false
let%test _ = compare (random [|3;4|]) (random [|3;4|]) = false


(*from a float list list to base_t*)
let of_list l= 
  let flat_l = List.flatten l in
  let n = List.length flat_l in
  let dim1 = List.length l in
  let dim2 = n  / dim1 in
  let res_raw = Genarray.create float64 c_layout [|n|] in
  for i = 0 to n - 1 do
    Genarray.set res_raw [|i|] (List.nth flat_l i)
  done;
  reshape res_raw [|dim1;dim2|]

let%test _ = compare (of_list [[0.;1.;2.];[3.;4.;5.]] ) (sequential [|2;3|]) = true
let%test _ = compare (of_list [[0.;1.;3.];[3.;4.;5.]] ) (sequential [|2;3|]) = false
let%test _ = compare (of_list [[0.;2.;2.];[3.;4.;5.]] ) (sequential [|2;3|]) = false

let to_arr x = 
  let flat_x = flatten x in
  let n      = numel x in
  let res =  Array.init n (fun x -> (get flat_x [|x|])) in
  res
  
let%test _ = to_arr (sequential [|2;2|]) = [|0.;1.;2.;3.|]

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
(*C: buffer    [100000]*)
(* res -> buffer*)
(* res_1 *)
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
(*float -> float -> float*)
(* Float.add*)
(* index -> float *)
(* init_nd *)
let element_wise_binary_caml input_1 input_2 ~map_func = 
  let s_1 = shape input_1 in
  let s_2 = shape input_2 in
  if s_1 <> s_2 then raise (Shape_error "element_wise_binary: shape not matched")
  else
    let output_shape = shape input_1 in
    let f = function index -> map_func (get input_1 index) (get input_2 index) in
init_nd output_shape f
(* init_nd [|2;2|] f*) 

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

let softmax ?(bt=CAML) tensor= 
  let ex = exp tensor ~bt in
  let sigma = sum_scalar ex in
  div_scalar ex sigma ~bt



let non_act ?(bt=CAML) =
  let f x =
    x in
  element_wise_unary ~map_func:f ~bt

(*loss function*)
let cross_entry  ?(bt=CAML) pre target =
  let ln_res = log pre in
  let mul_r  = mul target ln_res ~bt in
  mul_r |> sum' |> neg

(* matrix diagonal transpose *)
let transpose ?(bt=CAML) x = 
  let f index = [|index.(1);index.(0)|] in
  reindex x (reverse (shape x)) ~map_func:f ~bt

(* matrix subdiagonal transpose *)
(* i -> m - 1 - j *)
(* j -> n - 1 - i *)
let sub_transpose ?(bt=CAML) x =
  let s = shape x in
  let m = s.(0) in
  let n = s.(1) in
  let f index = 
    [|m-1-index.(1);n-1-index.(0)|] in
  reindex x (reverse s) ~map_func:f ~bt

(* reverse the element *)
let rev ?(bt=CAML) x = 
  x |> transpose ~bt|> sub_transpose ~bt

(* insert zero accroding to the stride *)
let insert_zero ?(_bt=CAML) x stride=
  let p = stride - 1 in
  let s = shape x in
  let m = s.(0) in
  let n = s.(1) in
  let r = m + (m-1)*p in
  let c = n + (n-1)*p in
  let f index = 
    let a = index.(0) in
    let b = index.(1) in
    if (a mod stride != 0) || (b mod stride != 0) then 0. 
    else (Genarray.get x [|a/stride;b/stride|]) in
  init_nd [|r;c|] f

(* merge a genarray array to a one-more dimentsion genarray *)
let merge ?(_bt=CAML) arr =
  let l = Array.length arr in
  let sub_shape = arr.(0) |> shape in
  let f i =
    if i == 0 then l else sub_shape.(i-1) in
  let s = Array.init ((sub_shape |> Array.length)+1) f in
  let f index =
    let sub_arr = arr.(index.(0)) in
    let sub_index = Array.init ((index|>Array.length)-1) (fun x -> index.(x+1)) in
    Genarray.get sub_arr sub_index in
init_nd s f 

(* split a genarray into a one-less dimension array *)
let split ?(_bt=CAML) tensor = 
  let s = tensor |> shape in
  let n = s.(0) in
  Array.init n (fun x -> (Genarray.slice_left tensor [|x|]))

let broad_cast ?(bt=CAML) input target_shape oxis = 
  let f input_index =
    let aux_f i =
      let aux_i_f x = if (x >= oxis) then x + 1 else x in
    input_index.(aux_i_f i) in
   Array.init ((Array.length input_index) - 1) aux_f in
reindex input target_shape ~map_func:f ~bt:bt

(* can be used as the example*)
(* slice a tensor by passing a tuple list*)
let  slice tensor slice_index = 
  let out_shape = Array.init (Array.length slice_index) (fun x -> Util.Misc.second slice_index.(x)) in 
  let map_func idx = Array.init (Array.length idx) (fun x -> (Util.Misc.first slice_index.(x)) + idx.(x)) in
  reindex tensor out_shape ~map_func

let sub_left ?(bt=CAML) tensor ofs len =
  let s = shape tensor in
  let out_shape = Array.init (Array.length s) (fun x -> match x with |0 -> len |_ -> s.(x)) in
  let map_func index = Array.init (Array.length index) (fun x -> match x with |0 -> (index.(0) + ofs)|_ -> index.(x) )in
  reindex tensor out_shape ~map_func ~bt

let pad_2d ?(constant=0.) tensor pd = 
  if pd == 0 then copy tensor
  else
          let s   = shape tensor in
          let r_t = s.(0) in
          let c_t = s.(1) in
          let new_s = Array.map (fun x -> x + pd + pd) s in
          let f index = 
            let r = index.(0) in
            let c = index.(1) in
            if (r >= pd && r < r_t+pd)&&(c >= pd && c < c_t + pd) then get tensor [|r-pd;c-pd|]
            else constant
          in init_nd new_s f

let pad_3d ?(constant=0.) tensor pd = 
  if pd == 0 then copy tensor
  else
          let s = shape tensor in
          let new_s = [|s.(0);s.(1)+2*pd;s.(2)+2*pd|] in
          let f index = 
            let d = index.(0) in
            let r = index.(1) in
            let c = index.(2) in
            if (r >= pd && r < s.(1)+pd)&&(c >= pd && c < s.(2) + pd) then get tensor [|d;r-pd;c-pd|]
            else constant
          in init_nd new_s f

let dot_sum ?(bt=CAML) x y =
  let mul_res = mul x y ~bt in
  sum_scalar mul_res

(* neural network operation *)
(* convolution without bias and bias*)
let conv2d ?(bt=CAML) ?(stride=1) ?(pd=0) tensor filter= 
  let tensor = pad_2d tensor pd in
  let shape_t = shape tensor in
  let r_t = shape_t.(0) in
  let c_t = shape_t.(1) in
  let shape_f = shape filter in
  let r_f = shape_f.(0) in
  let c_f = shape_f.(1) in
  let r = (r_t - r_f)/stride + 1 in
  let c = (c_t - c_f)/stride + 1 in
  let f index =
    [|stride*index.(0)+index.(2);stride*index.(1)+index.(3)|] in
  let xx = reindex tensor [|r;c;r_f;c_f|] ~map_func:f in
  let f index = 
    [|index.(2);index.(3)|] in
  let yy = reindex filter [|r;c;r_f;c_f|] ~map_func:f ~bt in
  let mul_res = mul xx yy in
  let f index = 
    [|index.(0);index.(1)|] in
  reindex_reduce mul_res [|r;c|] ~map_func:f ~bt
 
(* 3d convolution without bias and activation function*)
let conv3d ?(bt=CAML) ?(stride=1) ?(pd=0) tensor filter=
  let shape_t = shape tensor in
  let len_t   = shape_t.(0) in
  let f i = 
    let tensor_t = Genarray.slice_left tensor [|i|] in
    let filter_t = Genarray.slice_left filter [|i|] in
    conv2d tensor_t filter_t ~stride ~bt ~pd in
  let res_arr = Array.init len_t f in
  let init_val = zeros_like res_arr.(0) in
  Array.fold_left add init_val res_arr 

(*the layer operation of conv*)
let layer_conv3d_caml ?(bt=CAML) ?(stride=1) ?(pd=0) tensor filters= 
  let shape_f = shape filters in
  let filter_n = shape_f.(0) in
  let f i =
    let sub_filter = Genarray.slice_left filters [|i|] in
    conv3d tensor sub_filter ~stride ~bt ~pd in
  let res_arr = Array.init filter_n f in
  let f index = 
    let t = res_arr.(index.(0)) in
    Genarray.get t [|index.(1);index.(2)|] in
  let w = (res_arr.(0)|>shape).(0) in
  let h = (res_arr.(0)|>shape).(1) in
  init_nd [|filter_n;w;h|] f

(* boost type: the layer operation of conv *)
external c_layer_conv3d : base_t -> base_t -> int -> base_t  = "c_layer_conv3d"
let layer_conv3d_boost x y stride =
  copy (c_layer_conv3d x y stride)

(* the general api of conv3d *)
let layer_conv3d = layer_conv3d_boost

let max_pool2d ?(bt=CAML) ?(stride=1) tensor max_filter =
  let shape_t = shape tensor in
  let r_t = shape_t.(0) in
  let c_t = shape_t.(1) in
  let r_f = max_filter.(0) in
  let c_f = max_filter.(1) in
  let r   = (r_t - r_f)/stride + 1 in
  let c   = (c_t - c_f)/stride + 1 in
  let f index =
    [|stride * index.(0) + index.(2);stride * index.(1) + index.(3)|] in
  let xx = reindex tensor[|r;c;r_f;c_f|] ~map_func:f ~bt in
  let f index =
    let sub_tensor = slice xx [|(index.(0),1);(index.(1),1);(0,r_f);(0,c_f)|] in
    let sub_index  = max_i sub_tensor in
    let sub_array  = index_1d_nd sub_index [|r_f;c_f|] in
    let res_index  = [|index.(0);index.(1);sub_array.(0);sub_array.(1)|] in
        res_index in
  let res = reindex xx [|r;c|] ~map_func:f ~bt in
  res
  
let avg_pool2d ?(bt=CAML) ?(stride=1) tensor avg_filter = 
  let shape_t = shape tensor in
  let r_t = shape_t.(0) in
  let c_t = shape_t.(1) in
  let r_f = avg_filter.(0) in
  let c_f = avg_filter.(1) in
  let r   = (r_t - r_f)/stride + 1 in
  let c   = (c_t - c_f)/stride + 1 in
  let f index =
    [|stride * index.(0) + index.(2);stride * index.(1) + index.(3)|] in
  let xx = reindex tensor[|r;c;r_f;c_f|] ~map_func:f ~bt in
  let f index =
    let sub_tensor = slice xx [|(index.(0),1);(index.(1),1);(0,r_f);(0,c_f)|] in
    average_scalar sub_tensor in
  let res = init_nd [|r;c|] f in
  res

let avg_pool3d ?(bt=CAML) ?(stride=1) tensor avg_filter =
  let shape_t = shape tensor in
  let len     = shape_t.(0)  in 
  let f index = 
    let sub_tensor = Genarray.slice_left tensor [|index|] in
    avg_pool2d sub_tensor avg_filter ~bt ~stride in
  let res_arr = Array.init len f in
  let res_r   = (res_arr.(0)|>shape).(0) in
  let res_c   = (res_arr.(0)|>shape).(1) in
  let f index =
    Genarray.get res_arr.(index.(0)) [|index.(1);index.(2)|] in
  let res = init_nd [|len;res_r;res_c|] f in
  res

let max_pool3d ?(bt=CAML) ?(stride=1) tensor max_filter =
  let shape_t = shape tensor in
  let len     = shape_t.(0)  in 
  let f index = 
    let sub_tensor = Genarray.slice_left tensor [|index|] in
    max_pool2d sub_tensor max_filter ~bt ~stride in
  let res_arr = Array.init len f in
  let res_r   = (res_arr.(0)|>shape).(0) in
  let res_c   = (res_arr.(0)|>shape).(1) in
  let f index =
    Genarray.get res_arr.(index.(0)) [|index.(1);index.(2)|] in
  let res = init_nd [|len;res_r;res_c|] f in
  res

let flatten_conv tensor = 
  Bigarray.reshape tensor [|numel tensor;1|]

let dot ?(bt=CAML) x y = 
  let shape_b = 
    let shape_x = shape x in
    let shape_y = shape y in
    [|shape_x.(0);shape_x.(1);shape_y.(1)|] in
  let x_bd = broad_cast x shape_b 2 ~bt in
  let y_bd = broad_cast y shape_b 0 ~bt in
  let raw_res = mul x_bd y_bd ~bt in
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





(*misc *)
let print_vec x =
  print_endline "";
  let n = numel x in 
  for i = 0 to n - 1 do
    Printf.printf " %3g" (get x [|i|]);
  done;
  print_endline ""

let print_mat x = 
  print_endline "";
  let s = shape x in
  let r = s.(0) in
  let c = s.(1) in
  for i = 0 to r - 1 do
    for j = 0 to c - 1 do
      Printf.printf " %3g" (get x [|i;j|]);
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

let print_4d x_t =
  print_endline "";
  let s = shape x_t in
  let o = s.(0) in
  let p = s.(1) in
  let q = s.(2) in
  let r = s.(3) in
  for a = 0 to o - 1 do
    for b = 0 to p - 1 do
      for c = 0 to q - 1 do
        for d = 0 to r - 1 do
          Printf.printf " %g" (get x_t [|a;b;c;d|]);
        done;
      print_endline "";
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
  |4 -> print_4d x
  |_ -> raise (Shape_error "print:the shape is not supported")

let print_shape ?(prefix="") x=
  print_endline prefix;
  let s = shape x in
  let n = Array.length s in
  for i = 0 to n - 1 do
    Printf.printf "%d " s.(i)
  done;
  print_endline "";
  ()


let printf = Printf.printf 

let mat_dot ?(bt=CAML) x y  = 
  match !boost with
  |DEFAULT -> dot x y ~bt
  |_ -> copy (c_mat_mul x y !boost)

(* let%test _ =  compare (softmax (ones [|2;2|])) (ns 0.25 [|2;2|]) = true *)
(* let%test _ = print (conv2d (ones [|4;4|]) (ones [|2;2|]) ~stride:2 ~pd:1);false *)
(* let%test _ = print (conv3d (ones [|3;4;4|]) (ones [|3;2;2|]) ~stride:2 ~pd:1);false *)
(* let%test _ = print (layer_conv3d (ones [|3;4;4|]) (ones [|4;3;2;2|]) ~stride:2 ~pd:0);false *)
(* let%test _ = print (max_pool2d (sequential [|4;4|])  [|2;2|] ~stride:1);false *)
(* let%test _ = print (sequential [|3;4;4|]);false *)
(* let%test _ = print (max_pool3d (sequential [|3;4;4|])  [|2;2|] ~stride:1);false *)
