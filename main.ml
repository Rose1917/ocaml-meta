(* define the boost type. by default it will not boost *)
open Nn.Conv
open Op.Op_ad
include Util.Mnist

let if_match x y = 
  let x_i = max_i x in
  let y_i = max_i y in
  if x_i = y_i then 1 else 0

let _ = 
  (*set the boost : OMP_BOOST FMA_BOOST AVX_BOOST*)
  set_boost OMP_BOOST;  

  (* the defination of neural network *)
  (* let layer_1 = ref (conv ~tunnel:6 ~depth:1 ~width:5 ~height:5 ~stride:1 ~pad:2 ~act:tanh) in *)
  (* let layer_2 = ref (pool ~width:2 ~height:2 ~stride:2 ~pool_t:MeanPool) in *)
  (* let layer_3 = ref (conv ~tunnel:16 ~depth:6 ~width:5 ~height:5 ~stride:1 ~pad:0 ~act:tanh) in *)
  (* let layer_4 = ref (pool ~width:2 ~height:2 ~stride:2 ~pool_t:MeanPool) in *)
  (* let layer_5 = ref (conv ~tunnel:120 ~depth:16 ~width:5 ~height:5 ~stride:1 ~pad:0 ~act:tanh) in *)
  (* let layer_6 = ref (liner  84 120 sigmoid) in *)
  (* let layer_7 = ref (liner 10 84 softmax) in *)
  (* let test_network = init_net [|Conv !layer_1;Pool !layer_2;Conv !layer_3;Pool !layer_4; Conv !layer_5;Connect;Full !layer_6;Full !layer_7|] in *)

  (* let test_network = Nn.Net_marshal.load_net  "/home/march1917/recent/permanent/conv_network" in*)
  (* (* train the model*)*)
  (* let train_set,_,train_label = Util.Mnist.load_train_data() in*)
  (* let step = 0.01 in*)
  (* let _cycles = Util.Mnist.train_epoc in*)
  (* for _it = 50000 to 60000 - 1 do*)
  (*       (* Printf.printf "iteration %d\n" it; *)*)
  (*       (* flush_all(); *)*)
  (*       let input  = pack(reshape (slice train_set [|(_it,1);(0,784)|]) [|1;28;28|]) ~if_grad:false in*)
  (*       let target = pack (reshape (slice train_label [|(_it,1);(0,10)|]) [|10;1|]) in*)
  (*       let output = run_net input test_network in*)

  (*       Util.Mnist.draw_image !(primal input);*)
  (*       print output;*)
  (*       Printf.printf "the prediction is %d\n" (max_i output);*)

  (*       let loss   = cross_entry output target in*)
  (*       diff loss;*)
  (*       train loss step;*)
  (*  done;*)
  (*  Nn.Net_marshal.save_net test_network "/home/march1917/recent/permanent/conv_network";*)

   let test_network = Nn.Net_marshal.load_net  "./permanent/conv_network" in

  (* test the model *)
  let match_cnt = ref 0 in 
  let _cycles = test_epoc in 
  let test_set,_,test_label = load_test_data () in 
  Printf.printf "test iteration begin,cycles %d\n" _cycles; 
  Stdlib.print_endline "iteration begin"; 
  for it = 0 to 10000 - 1 do 
    let input = pack (reshape (slice test_set [|(it,1);(0,784)|]) [|1;28;28|]) ~if_grad:false  in 
    let target = pack (reshape (slice test_label [|(it,1);(0,10)|]) [|10;1|])  in  
    let z =  run_net input test_network in 
    (* draw_image(input|> unpack |> (!)); *) 
    (* Printf.printf "the res is %d\n" (max_i target); *) 
    match_cnt:= !match_cnt + (if_match target z); 
  done;  
  let _a = Printf.printf "%d matched the accuray is %g%%\n" !match_cnt ((Float.of_int !match_cnt)/. 100.) in 
  () 
