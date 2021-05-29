open Var.D
(*obtain the basic property of the ad tensor*)
       
let primal x = x.base_val
let deri x = x.deri_val
let adj x = x.adj_fun
let if_trained x = x.if_trained
let if_grad x = x.if_grad

(*some translation between the base and ad ndarray*)
let create ?(if_grad=true) base deri f = {
  base_val= ref base;
  deri_val= ref deri;
  adj_fun=f;
  if_trained = ref false;
  if_grad = ref if_grad;
}

let pack ?(if_grad=true) tensor = 
  let deri = Op_base.zeros_like tensor in
  let adj_fun _ca l = l in
  create ~if_grad:if_grad tensor deri adj_fun

let unpack ad_tensor = 
  primal ad_tensor
(*the basic creation operator*)

  


open Op_base
(*create functions*)
let sequential ?(if_grad=true) ?(a = 0.) ?(step = 1.0) shape = 
  let x = sequential ~a ~step shape in
  pack ~if_grad:if_grad x

let random ?(if_grad=true) ?(bound = 1.0) shape = 
  let x = random ~bound shape in
  pack ~if_grad x

let zeros ?(if_grad=true) shape = 
  let x = zeros shape in
  pack ~if_grad x

let ones ?(if_grad=true) shape = 
  let x = ones shape in
  pack ~if_grad x

let ns ?(if_grad=true) n  shape = 
  let x = ns n shape in
  pack ~if_grad x

(*the basic ndarray operation*)
let add ?(bt=CAML) x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let y' = primal y in
  let base_val = add !x' !y' ~bt in

  let adj_fun ca t =
    let r = mul (ones_like !x') !ca ~bt in
    let s = mul (ones_like !y') !ca ~bt in
    (r,x)::(s,y)::t in
  create base_val (zeros_like base_val) adj_fun

let sub ?(bt=CAML) x y =
  (*calculate the primal value*)
  let x' = primal x in
  let y' = primal y in
  let base_val = sub !x' !y' ~bt in

  let adj_fun ca t =
    let r = mul (ones_like !x') !ca ~bt in
    let s = mul (neg((ones_like !y')) ~bt ) !ca ~bt in
    (r,x)::(s,y)::t in
  create base_val (zeros_like base_val) adj_fun

let mul ?(bt=CAML) x y =
  (*calculate the primal value*)
  let x' = primal x in
  let y' = primal y in
  let base_val = mul !x' !y' ~bt in

  let adj_fun ca t =
    let r = mul !y' !ca ~bt in
    let s = mul !x' !ca ~bt in
    (r,x)::(s,y)::t in
  create base_val (zeros_like base_val) adj_fun

let div ?(bt=CAML)  x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let y' = primal y in
  let base_val = div !x' !y' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (reci_procal !y' ~bt) !ca ~bt in
    let s = Op_base.mul (Op_base.mul !x' (reci_procal (neg (sqr !y' ~bt ) ~bt ) ~bt) ~bt) !ca ~bt in
    (r,x)::(s,y)::t in
  create base_val (zeros_like base_val) adj_fun

let add_scalar ?(bt=CAML) x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = add_scalar !x' y ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.ones_like !x') !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let sub_scalar ?(bt=CAML) x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = sub_scalar !x' y ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.ones_like !x') !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let mul_scalar ?(bt=CAML) x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = mul_scalar !x' y ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.ns_like y !x') !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let div_scalar ?(bt=CAML) x y = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = div_scalar !x' y ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (reci_procal (Op_base.ones_like  !x') ~bt) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun


let sin ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = sin !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.cos !x' ~bt) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let cos ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = cos !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.neg (Op_base.sin !x' ~bt) ~bt) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let sqr ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = sqr !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.mul_scalar !x' 2. ~bt) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let sum ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = sum' !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul_scalar (Op_base.ones_like !x') (Op_base.get !ca [|0|]) ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

(*some activation functions*)
let sigmoid  ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = Op_base.sigmoid !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.mul base_val (Op_base.add_scalar (Op_base.neg (base_val) ~bt) 1. ~bt) ~bt) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let relu ?(bt=CAML) x =
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = relu !x' ~bt in

  let adj_fun ca t = 
    let r = Op_base.mul (base_val) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let tanh ?(bt=CAML) x =
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = tanh !x' ~bt in

  let adj_fun ca t = 
    let r = Op_base.mul (Op_base.add_scalar (Op_base.neg (Op_base.sqr base_val ~bt) ~bt) 1. ~bt)!ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let softmax ?(bt=CAML) x =
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = softmax !x' ~bt in

  let adj_fun ca t = 
    let x_val = Util.Misc.non_zero (Op_base.to_arr !ca) in
    let x_inx = Util.Misc.non_zeroi (Op_base.to_arr !ca) in
    let f index = 
      if index.(0) = x_inx then (x_val *. (get base_val index) *. (1. -. (get base_val index)))
      else -.(x_val *. (get base_val [|x_inx;0|]) *. (get base_val index)) in
    let r = Op_base.init_nd (shape !x') f in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun

let cross_entry ?(bt=CAML) pre target =
  let pre' = primal pre in
  let tar' = primal target in
  let base_val = cross_entry !pre' !tar' in

  let adj_fun ca t = 
    let s = get !ca [|0|] in
    let r = Op_base.mul_scalar (Op_base.mul (neg !tar' ~bt) (reci_procal !pre' ~bt)) s ~bt in
    (r,pre)::t in
  create base_val (zeros_like base_val) adj_fun

let non_act ?(bt=CAML) x =
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = !x' in

  let adj_fun ca t = 
    let r = Op_base.mul (Op_base.ones_like base_val) !ca ~bt in
    (r,x)::t in
  create base_val (zeros_like base_val) adj_fun


let mat_mul ?(bt=CAML) x y= 
  let x' = primal x in
  let y' = primal y in
  let base_val = Op_base.mat_dot !x' !y' ~bt in

  let adj_fun ca t =
    let r = Op_base.mat_dot (!ca) (Op_base.transpose(!y') ~bt) ~bt in
    let s = Op_base.mat_dot (Op_base.transpose(!x') ~bt) !ca ~bt in
    (r,x)::(s,y)::t in
  create base_val (zeros_like base_val) adj_fun

let pad_2d ?(bt=CAML) ?(constant=0.) tensor pd =
  let tensor' = primal tensor in
  let base_val = Op_base.pad_2d !tensor' pd ~constant in

  let adj_func ca t = 
    let map_func index = 
      Array.map (fun x -> x + pd) index in
    let r = Op_base.reindex !ca (shape !tensor') ~map_func ~bt in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let pad_3d ?(bt=CAML) ?(constant=0.) tensor pd =
  let tensor' = primal tensor in
  let base_val = Op_base.pad_3d !tensor' pd ~constant in

  let adj_func ca t =
    let map_func index =
      [|index.(0);index.(1)+pd;index.(2)+pd|] in
    let r = Op_base.reindex !ca (shape !tensor') ~map_func ~bt in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func


let conv2d ?(bt=CAML) ?(stride=1)  tensor filter= 
  let tensor' = primal tensor in
  let filter' = primal filter in
  let base_val = Op_base.conv2d !tensor' !filter' ~stride ~bt in

  let adj_func ca t =
    let z_ca  = Op_base.insert_zero !ca stride in
    let pad_t = Op_base.pad_2d z_ca ((shape !filter').(0)-1) in
    let r     = Op_base.conv2d pad_t (Op_base.rev !filter' ~bt) ~bt  in
    let s     = Op_base.conv2d !tensor' z_ca  ~bt in
    (r,tensor)::(s,filter)::t in
  create base_val (zeros_like base_val) adj_func
    
let conv3d ?(bt=CAML) ?(stride=1)  tensor filter=
  let tensor' = primal tensor in
  let filter' = primal filter in
  let base_val = Op_base.conv3d !tensor' !filter' ~stride ~bt in

  let adj_func ca t = 
    let tensor_arr = Op_base.split !tensor' in
    let filter_arr = Op_base.split !filter' in
    let z_c = Op_base.insert_zero !ca stride in
    let pad_t = Op_base.pad_2d z_c ((shape filter_arr.(0)).(0)-1) in
    let f_r i =
      let f_t = filter_arr.(i) in
      Op_base.conv2d pad_t (Op_base.rev f_t ~bt) ~bt in
    let r_arr = Array.init (Array.length tensor_arr) f_r in
    
    let f_s i =
      let t_t = tensor_arr.(i) in
      Op_base.conv2d t_t z_c ~bt in
    let s_arr = Array.init (Array.length tensor_arr) f_s in
  (Op_base.merge(r_arr),tensor)::(Op_base.merge(s_arr),filter)::t in
  create base_val (zeros_like base_val) adj_func

let layer_conv3d ?(bt=CAML) ?(stride=1) tensor filter=
  let tensor' = primal tensor in
  let filter' = primal filter in
  let base_val = Op_base.layer_conv3d !tensor' !filter' stride  in

  (* ca -> tensor -> filter -> r_tensor *)
  let get_r c f =
      let f_arr = Op_base.split f in
      let z_c = Op_base.insert_zero c stride in
      let pad_t = Op_base.pad_2d z_c ((shape f_arr.(0)).(0)-1) in
      let f i =
        let f_t = f_arr.(i) in
        Op_base.conv2d pad_t (Op_base.rev f_t ~bt) ~bt in
      Op_base.merge (Array.init (Array.length f_arr) f) in

  (*ca -> filter -> s_filter *)
  let get_s c t =
     let z_c = Op_base.insert_zero c stride in
     let t_arr = Op_base.split t in
     let f i =
       let t_t = t_arr.(i) in
       Op_base.conv2d t_t z_c ~bt in
     Op_base.merge (Array.init (Array.length t_arr) f) in

  let adj_func ca t=
    let ca_arr = Op_base.split !ca in
    let filter_arr = Op_base.split !filter' in
    let f i =
      let sub_ca = ca_arr.(i) in
      let sub_filter = filter_arr.(i) in
      get_r sub_ca sub_filter in
      (* i -> sub_ca -> sub_filter -> tensor -> sub_r *)
    let r_arr = Array.init (Array.length filter_arr) f in
    let r = Array.fold_left (Op_base.add) (zeros_like !tensor') r_arr in

    let f i =
      let sub_ca = ca_arr.(i) in
      get_s sub_ca !tensor' in
    let s_arr = Array.init (Array.length filter_arr) f in
    let s = Op_base.merge s_arr in
    (r,tensor)::(s,filter)::t
  in
create base_val (zeros_like base_val) adj_func


external deri_filter : base_t -> base_t -> int -> base_t  = "c_deri_filter"
external deri_tensor : base_t -> base_t -> int -> base_t  = "c_deri_tensor"

let layer_conv3d_boost ?(stride=1) tensor filter=
  let tensor' = primal tensor in
  let filter' = primal filter in
  let base_val = Op_base.layer_conv3d !tensor' !filter' stride  in

  let adj_func ca t=
    let r = Op_base.copy (deri_tensor !ca !filter' stride) in
    let s = Op_base.copy (deri_filter !ca !tensor' stride) in
    (r,tensor)::(s,filter)::t
  in
create base_val (zeros_like base_val) adj_func


let max_pool2d ?(bt=CAML) ?(stride=1) tensor max_filter = 
  let tensor' = primal tensor in
  let base_val = Op_base.max_pool2d !tensor' max_filter ~stride ~bt in

  let adj_func ca t = 
    let f index = 
      let x = index.(0) / max_filter.(0) in
      let y = index.(1) / max_filter.(1) in
      let t_val = Bigarray.Genarray.get !tensor' index in
      let max_v = Bigarray.Genarray.get base_val [|x;y|] in
      if t_val <> max_v then 0. else (Bigarray.Genarray.get !ca [|x;y|]) in
    let r = Op_base.init_nd  (shape !tensor') f in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let max_pool3d ?(bt=CAML) ?(stride=1) tensor max_filter= 
  let tensor' = primal tensor in
  let base_val = Op_base.max_pool3d !tensor' max_filter ~stride ~bt in

  let adj_func ca t = 
  let f index = 
    let x = index.(0) in
    let y = index.(1) / max_filter.(0) in
    let z = index.(2) / max_filter.(1) in
    let t_val = Bigarray.Genarray.get !tensor' index in
    let max_v = Bigarray.Genarray.get base_val [|x;y;z|] in
    if t_val <> max_v then 0. else (Bigarray.Genarray.get !ca [|x;y;z|]) in
  let r = Op_base.init_nd (shape !tensor') f in
  (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let avg_pool3d ?(bt=CAML) ?(stride=1) tensor max_filter= 
  let tensor' = primal tensor in
  let base_val = Op_base.avg_pool3d !tensor' max_filter ~stride ~bt in

  let adj_func ca t = 
  let f index = 
    let x = index.(0) in
    let y = index.(1) / max_filter.(0) in
    let z = index.(2) / max_filter.(1) in
    let t = Bigarray.Genarray.get !ca [|x;y;z|] in
    t *. 0.25 in
  let r = Op_base.init_nd (shape !tensor') f in
  (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let broad_conv ?(bt=CAML) tensor like =
  let like' = primal like in
  let s = shape !like' in
  let tensor' = primal tensor in
  let f index =
    [|index.(0);0|] in
  let base_val = Op_base.reindex !tensor' s ~map_func:f ~bt in

  let adj_func ca t = 
    let r = Op_base.reindex_reduce !ca (shape !tensor') ~map_func:f in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let flatten ?(_bt=CAML) tensor = 
  let tensor' = primal tensor in
  let base_val = Op_base.flatten_conv !tensor'  in

  let adj_func ca t =
    let r = Op_base.reshape !ca (Op_base.shape !tensor')  in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

let flatten_conv ?(_bt=CAML) tensor = 
  let tensor' = primal tensor in
  let base_val = Op_base.flatten_conv !tensor'  in

  let adj_func ca t =
    let r = Op_base.reshape !ca (Op_base.shape !tensor')  in
    (r,tensor)::t in
  create base_val (zeros_like base_val) adj_func

(*some util functions*)
let get_ele x index= 
  get (!(primal x)) index

let max_i x = 
  x|>primal |>(!)|>Op_base.to_arr|>Util.Misc.max_i

let get_grad x index = 
  get (!(deri x)) index

let print_primal ?(prefix="") x = 
  print (!(primal x)) ~prefix:("primal "^prefix)

let print_grad ?(prefix="") x = 
  print (!(deri x)) ~prefix:("deri "^prefix)

let print_endline x =
  Stdlib.print_endline x

let print_shape x= 
  print_endline ""; 
  let s = x|>primal|>(!)|>Op_base.shape in
  let n = Array.length s in
  for i = 0 to n - 1 do
    Printf.printf "%d " s.(i)
  done;
  print_endline "";
  ()


let print ?(prefix="") x = 
  print_primal x ~prefix;
  print_grad x ~prefix

let rec back_pro ?(bt=CAML) xs = 
  match xs with 
  | [] -> ()
  |(v,dr)::t ->
    let aa = deri dr in
    let adjfun = adj dr in
    let if_trained = if_trained dr in
    aa:= Op_base.add !aa v ~bt;
    if_trained := false ;

    let stack = adjfun aa t in
    back_pro stack 
let diff z = 
  back_pro [((Op_base.ones_like !(deri z)),z)]

let rec  update ?(bt=CAML) xs step =
  match xs with 
  |[] -> () 
  |(_v,dr)::t ->
    let pp = primal dr in
    let adjfun = adj dr in
    let aa = deri dr in
    let i = if_trained dr in
    let g = if_grad dr in
    if (!i = true || !g = false) then () else pp:= Op_base.sub !pp (Op_base.mul_scalar !aa step ~bt) ~bt;i:=true;aa := (Op_base.zeros_like !aa); 
    let stack = adjfun aa t in
    update stack step

let train z step = 
  update [((Op_base.ones_like !(deri z)),z)] step

let set_boost x = 
  set_boost x
