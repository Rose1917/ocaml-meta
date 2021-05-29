include Var.D
include Var.Nn
include Op.Op_ad

(* the exception defination *)
exception Unrecognized_layer of string

(* the basic datastructure for marshal *)
type var_mar = Var.D.base_t

type act_type =
  |Sigmoid
  |Tanh
  |Relu
  |Softmax
  |NonAct

(* since marshal does NOT store the type information,so we need to store the layer type explictly *)
type layer_type =
  |FullType
  |ConvType
  |ConnectType
  |PoolType

type full_layer_mar = {
  w:var_mar;
  b:var_mar;
  a:act_type
}

type conv_layer_mar = {
  w:var_mar;
  b:var_mar;
  stride:int;
  pad:int;
  a:act_type;
}

type pool_layer_mar = {
  h:int;
  w:int;
  stride:int;
  t:pool_type;
}

type layer_mar = 
  |Fullm of full_layer_mar
  |Convm of conv_layer_mar
  |Poolm of pool_layer_mar
  |Connectm
  

type net_mar = {
  layers:(layer_mar*layer_type) array
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

(* transfer the activation variant to the correponding function *)
let from_act a = 
  match a with 
  |Sigmoid -> sigmoid
  |Relu -> relu
  |Tanh -> tanh 
  |Softmax -> softmax
  |NonAct -> non_act


(* transfer the raw full layer to the marshal full layer *)
let to_marshal_full (f:full_layer) :full_layer_mar = 
  {
    w = f.w |> primal |> (!);
    b = f.b |> primal |> (!);
    a = f.act |> to_act;
  } 


(* transfer the marshal full layer to the raw full layer *)
let from_marshal_full (f:full_layer_mar) :full_layer =
  {
    w = f.w |> pack;
    b = f.b |> pack;
    act = f.a |> from_act;
  }

(* transfer the raw conv layer to the marshal conv layer *)
let to_marshal_conv (f:conv_layer) :conv_layer_mar = 
  {
    w = f.w |> primal |> (!);
    b = f.b |> primal |> (!);
    stride = f.stride;
    pad   = f.pad;
    a = f.act |> to_act;
  } 

(* transfer the marshal conv layer to the raw conv layer *)
let from_marshal_conv (f:conv_layer_mar) :conv_layer =
  {
    w = f.w |> pack;
    b = f.b |> pack;
    stride = f.stride;
    pad    = f.pad;
    act = f.a |> from_act;
  }


(* transfer the raw pool layer to the marshal pool layer *)
let to_marshal_pool (f:pool_layer) :pool_layer_mar = 
  {
    h = f.h;
    w = f.w;
    stride = f.stride;
    t = f.t;
  } 


(* transfer the marshal conv layer to the raw conv layer *)
let from_marshal_pool (f:pool_layer_mar) :pool_layer =
  {
    h = f.h;
    w = f.w;
    stride = f.stride;
    t = f.t;
  }

(* transfer the a layer to non-functional layer*)
let to_marshal_lay (l:layer) :(layer_mar*layer_type) = 
  match l with
  |Full f -> (Fullm (to_marshal_full f),FullType)
  |Conv c -> (Convm (to_marshal_conv c),ConvType)
  |Pool p -> (Poolm (to_marshal_pool p),PoolType)
  |Connect ->((Connectm),ConnectType)

(* tranfer the non_function layer to the raw layer *)
let from_marshal_lay (l:(layer_mar*layer_type)):layer = 
  match l with
  |((Fullm f),FullType) -> (Full (from_marshal_full f))
  |((Convm c),ConvType) -> (Conv (from_marshal_conv c))
  |((Poolm p),PoolType) -> (Pool (from_marshal_pool p))
  |((Connectm),ConnectType) -> Connect
  |_ -> raise (Unrecognized_layer "reading from the file error:unrecogized layer type")


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
