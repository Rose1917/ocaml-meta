include Var.Nn
include Base.Op_base

open Base.Op_ad
let run_layer ?(bt=CAML) input x = 
  let mul_res = mat_mul x.w input ~bt in
  let add_res = add mul_res x.b ~bt in
  x.acv add_res 
let run_net ?(bt=CAML) x nn =
  Array.fold_left (run_layer ~bt) x nn.layers
