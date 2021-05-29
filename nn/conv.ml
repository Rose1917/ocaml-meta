include Var.Nn
include Op.Op_base

open Op.Op_ad

(* create a FC liner *)
let liner  w_r w_c act:full_layer =
{ 
    w = random  [|w_r;w_c|] ~bound:0.1;
    b = zeros [|w_r;1|] ;
    act = act;
}

(* create a conv layer *)
let conv  ~tunnel ~depth ~width ~height ~stride ~pad ~act = {
  w = random [|tunnel;depth;width;height|] ~bound:0.01;
  b = zeros [|tunnel;1|] ;
  pad = pad;
  stride = stride;
  act = act
}

(* create a list according to a layer list *)
let init_net (x:layer array) = 
  {
    layers = x;
  }

(* create a pool layer*)
let pool ~width ~height ~stride ~pool_t= {
  h=height;
  w=width;
  stride=stride;
  t=pool_t;
}

let connect () =
  Connect
let run_layer ?(bt=CAML) input x = 
  match x with 
  |(Connect) -> flatten input
  |(Pool p) -> if p.t == MaxPool then max_pool3d input [|p.h;p.w|] ~stride:p.stride ~bt
               else avg_pool3d input [|p.h;p.w|] ~stride:p.stride ~bt
  |(Full f) ->f.act (add (mat_mul f.w input ~bt) f.b ~bt) ~bt
  |(Conv c) ->
    let pd_res = pad_3d input c.pad in
    let conv_res = layer_conv3d_boost pd_res c.w ~stride:c.stride  in
    let b = broad_conv c.b conv_res in
    let add_res = add b conv_res in
    c.act add_res


let run_net ?(bt=CAML) x nn =
  Array.fold_left (run_layer ~bt) x nn.layers
