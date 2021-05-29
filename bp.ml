(* define the boost type. by default it will not boost*)
open Nn.Full
open Op.Op_ad
open Util.Mnist

let if_match x y = 
  let x_i = max_i x in
  let y_i = max_i y in
  if x_i = y_i then 1 else 0

let _ = 
  (*set the boost : OMP_BOOST FMA_BOOST AVX_BOOST*)
  set_boost OMP_BOOST;  

  (*
  let layer_1 = ref (liner 40 (28*28) sigmoid) in
  let layer_2 = ref (liner 10 40 softmax) in
  let test_network = init_net [|!layer_1;!layer_2;|] in

  (*the neural network defination*)

  (*prepare for the neural network arguments,since we do not need the d(loss)/d(input),so we set the if_grad = false*)
  (*for convinence,we set the arguments sequentially*)

  (*iterate to train,every diff function will do a forward propogation and backward propogation*)
  (*train function will do a update according to the gradient set before by diff function*)

  let train_set,_,train_label = load_train_data() in 
  let step = 0.05 in
  let _cycles = train_epoc in
  let epoch  = int_of_string Sys.argv.(1) in
  for _   = 0 to epoch - 1 do 
    for it = 0 to _cycles - 1 do
      let input  = pack (reshape (slice train_set [|(it,1);(0,784)|]) [|(28*28);1|])~if_grad:false in
      let target = pack (reshape (slice train_label [|(it,1);(0,10)|]) [|10;1|]) in
      let output =  run_net input test_network in
      let loss   = cross_entry output target in
      diff loss;
      train loss step;
    done; 
  done;
  Nn.Net_marshal.save_net test_network "/home/march1917/recent/permanent/test_network";
  *)
  let test_network = Nn.Net_marshal.load_net "/home/march1917/recent/permanent/test_network" in
  let match_cnt = ref 0 in
  let _cycles = test_epoc in
  let test_set,_,test_label = load_test_data () in
  Printf.printf "test iteration begin,cycles %d\n" _cycles;
  Stdlib.print_endline "iteration begin";
  for it = 0 to _cycles - 1 do
    let input = pack (reshape (slice test_set [|(it,1);(0,784)|]) [|(28*28);1|]) ~if_grad:false in
    let target = pack (reshape (slice test_label [|(it,1);(0,10)|]) [|10;1|])  in 
    let z =  run_net input test_network in
    draw_image(input|> unpack |> (!));
    Printf.printf "the res is %d\n" (max_i target);
    match_cnt:= !match_cnt + (if_match target z);
  done; 
  let _a = Printf.printf "%d matched the accuray is %g\n" !match_cnt ((Float.of_int !match_cnt)/. 100.) in
  ()
