(* define the boost type. by default it will not boost*)
open Nn.Full
open Op.Op_ad
let _ = 
  (*set the boost : OMP_BOOST FMA_BOOST AVX_BOOST*)
  set_boost OMP_BOOST;
  
  let layer_1 = liner 20 1 relu in
  let layer_2 = liner 70 20 relu in
  let layer_3 = liner 1 70 non_act in
  let test_network = init_net [|layer_1;layer_2;layer_3|] in

  (*the neural network defination*)

  (*prepare for the neural network arguments,since we do not need the d(loss)/d(input),so we set the if_grad = false*)
  (*for convinence,we set the arguments sequentially*)

  (*iterate to train,every diff function will do a forward propogation and backward propogation*)
  (*train function will do a update according to the gradient set before by diff function*)

  let step = 0.01 in
  let cycles = 100 in
  Printf.printf "train iteration begin,cycles %d ,step %f\n" cycles step;
  Stdlib.print_endline "iteration begin";
  for it = 1 to cycles do
    let input =  (random ~if_grad:false [|1;1|]) in
    let target = add_scalar (sqr input) 1. in
    
    let z =  run_net input test_network in
    let loss = sum (sqr (sub z target)) in
    diff loss;

    let loss_float = get_ele loss [|0|] in
    Printf.printf "iteration %d,loss %g\n" it loss_float;
    train z step;
  done ;

  let mean_loss = ref 0. in
  let cycles = 100 in
  Printf.printf "test iteration begin,cycles %d\n" cycles;
  Stdlib.print_endline "iteration begin";
  for it = 1 to cycles do
    let input =  (random ~if_grad:false ~bound:100. [|1;1|]) in
    let target = add_scalar (sqr input) 1. in
    
    let z =  run_net input test_network in
    let loss = sum (sqr (sub z target)) in
    let loss_float = get_ele loss [|0|] in
    mean_loss := !mean_loss +. loss_float;
    Printf.printf "iteration %d,loss %g\n" it loss_float;
  done; 
  
  
  Printf.printf "test iteration over,cycles %d\n" cycles;
  Printf.printf "the mean loss is  %g\n" (!mean_loss /. Float.of_int(cycles));
