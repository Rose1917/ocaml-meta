include Var.D
include Var.Nn
include Op.Op_ad
(*the datastructure for marshal*)
type var_mar = Var.D.base_t

type act_type =
  |Sigmoid
  |Tanh
  |Relu
  |Softmax
  |NonAct

type layer_mar = {
  w:var_mar;
  b:var_mar;
  a:act_type
}

type net_mar = {
  layers:layer_mar array
}

(* transfer the activation function to the correponding vairant *)
let to_act a = 
  let _sd = sigmoid in
  let _ru = relu in
  let _th = tanh in
  let _so = softmax in
  let _na = non_act in

  if      a == _sd then Sigmoid
  else if a == _ru then Relu
  else if a == _th then Tanh
  else if a == _so then Softmax
  else NonAct
let from_act a = 
  match a with 
  |Sigmoid -> sigmoid
  |Relu -> relu
  |Tanh -> tanh 
  |Softmax -> softmax
  |NonAct -> non_act

(* transfer the a layer to non-functional layer*)
let to_marshal_lay (l:layer) :layer_mar = 
  {
        w = l.w |> (Op.Op_ad.unpack) |> (!);
        b = l.b |> (Op.Op_ad.unpack) |> (!);
        a = to_act l.acv
  }

(* tranfer the non_function layer to the source layer *)
let from_marshal_lay (l:layer_mar):layer = 
  {
    w = Op.Op_ad.pack l.w ;
    b = Op.Op_ad.pack l.b;
    acv = from_act l.a
  }


(* transfer a net to a non-function net *)
let to_marshal_net (n:network):net_mar = 
  let src = n.layers in
  let len = Array.length src in
  let map_func i = 
    src.(i) |> to_marshal_lay in
  {
  layers = Array.init len map_func
  }

(* transfter a non-functional net back *)
let from_marshal_net (n:net_mar) :network = 
  let src = n.layers in
  let len = Array.length src in
  let map_func i =
    src.(i) |> from_marshal_lay in
  {
    layers = Array.init len map_func
  }

(* save a net *)
let save_net (n:network) file_path = 
  let oo = open_out_bin file_path in
  Marshal.to_channel oo (to_marshal_net n) [];
  close_out oo

(*load a net *)
let load_net file_path = 
  let oi = open_in_bin file_path in
  let mar_net = (Marshal.from_channel oi :net_mar) in
  from_marshal_net mar_net
