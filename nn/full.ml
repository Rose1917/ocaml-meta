include Var.Nn
include Op.Op_base

open Op.Op_ad
let liner  w_r w_c act =
{ 
    w = random  [|w_r;w_c|] ~bound:0.1;
    b = random [|w_r;1|] ~bound:0.1;
    acv = act;
}

let init_net x = 
  {
    layers = x;
  }

let run_layer ?(bt=CAML) input x = 
  let mul_res = mat_mul x.w input ~bt in
  let add_res = add mul_res x.b ~bt in
  x.acv add_res 
let run_net ?(bt=CAML) x nn =
  Array.fold_left (run_layer ~bt) x nn.layers
