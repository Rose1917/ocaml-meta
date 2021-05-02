open Owl_dense_ndarray_d
(*the create operator*)
let sequential shape = 
  sequential shape
let zeros shape =
  zeros shape
let ones shape =
  ones shape
let zeros_like tensor =  
  let s = shape tensor in 
  (zeros s)
let ones_like tensor =
  let s = shape tensor in 
  (ones s)
  
(*The basic binary  ndarray operation *)

let add x y =
  add x y
let sub x y =
  sub x y
let mul x y =
  mul x y
let div x y =
  div x y
(*the basic unary operation*)
let abs x = 
  abs x
let neg x = 
  neg x
let floor x = 
  floor x
let sqr x =
  sqr x
let sqrt x = 
  sqrt x
let ln x =
  log x
let log2 x =
  log2 x
let log10 x = 
  log10 x
let exp x =
  exp x
let sin x =
  sin x
let cos x =
  cos x
let tan x =
  tan x
let sum x = 
  sum x
let reci_procal x =
  div_scalar x 1.
let sigmoid x = 
  sigmoid x
(*The basic operation between the ndarray and scalar*)
let add_scalar x y =
  add_scalar x y
let sub_scalar x y =
  sub_scalar x y
let mul_scalar x y = 
  mul_scalar x y
let div_scalar x y =
  div_scalar x y

(*The operation between scalar and scalar*)
let scalar_add x y =
  scalar_add x y
let scalar_sub x y = 
  scalar_sub x y
let scalar_mul x y = 
  scalar_sub x y
let scalar_div x y = 
  scalar_div x y

(*operation from ndarray to scalar*)
let sum' x = 
  sum' x

  


(*the generic property operation*)
let shape x = 
  shape x
let numel x =
  numel x
let get x index=
  get x index

(* other util function *)
let print x =
  print x

open Owl_dense_matrix_d 
(*matrix operation*)
let mat_dot x y = 
  dot x y
let transpose x =
  transpose x 

 
