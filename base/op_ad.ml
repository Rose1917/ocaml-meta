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


let sigmoid  ?(bt=CAML) x = 
  (*calculate the primal value*)
  let x' = primal x in
  let base_val = Op_base.sigmoid !x' ~bt in

  let adj_fun ca t =
    let r = Op_base.mul (Op_base.mul base_val (Op_base.add_scalar (Op_base.neg (base_val) ~bt) 1. ~bt) ~bt) !ca ~bt in
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

(*some util functions*)
let get_ele x index= 
  get (!(primal x)) index

let get_grad x index = 
  get (!(deri x)) index

let print_primal x = 
  print (!(primal x))

let print_grad x = 
  print (!(deri x))

let print_endline x =
  Stdlib.print_endline x


let print x = 
  print (!(primal x));
  print (!(deri x))

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
let diff f a b c d e g  = 
  let z = f a b c d e g in
  back_pro [((Op_base.ones_like !(deri z)),z)];
  z

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
