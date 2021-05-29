include Op.Op_ad
(* let%test "conv_2d"= *) 
(*   let input = sequential [|4;4|] in *)
(*   print input ~prefix:"input"; *)
(*   let filter = sequential [|2;2|] in *)
(*   let o = conv2d input filter ~stride:2 in *)
(*   print filter ~prefix:"filter"; *)
(*   print o ~prefix:"output"; *)
(*   (1* let max_res = max_pool2d o [|2;2|] ~stride:2 in *1) *)
(*   (1* print max_res ~prefix:"max_res"; *1) *)
(*   diff o; *)

(*   print input ~prefix:"input"; *)
(*   print filter ~prefix:"filter"; *)
(*   print o ~prefix:"output"; *)
(*   (1* print max_res ~prefix:"max_res"; *1) *)
(*   false *)
  


(* let%test "max_pool3d"= *) 
(*   let input = random [|2;14;14|] in *)
(*   print input ~prefix:"input"; *)
(*   let max_res = max_pool3d input [|2;2|] ~stride:2 in *)
(*   print max_res ~prefix:"max_res"; *)
(*   diff max_res; *)

(*   print input ~prefix:"input(after diff)"; *)
(*   print max_res ~prefix:"max_res(after diff)"; *)
(*   false *)
  


(* let%test "conv_3d"= *) 
(*   let input = sequential [|3;4;4|] in *)
(*   print input ~prefix:"input"; *)
(*   let conv_core = ones [|3;2;2|] in *)
(*   print conv_core ~prefix:"conv core"; *)
(*   let conv_res = conv3d input conv_core ~stride:2 in *)
(*   print conv_res ~prefix:"conv_res"; *)
(*   diff conv_res; *)

(*   print input ~prefix:"input"; *)
(*   print conv_res ~prefix:"conv_res"; *)
(*   false *)
  



(* let%test "layer_conv_3d"= *) 
(*   let input = sequential [|3;4;4|] in *)
(*   print input ~prefix:"input"; *)
(*   let conv_core = ones [|4;3;2;2|] in *)
(*   print conv_core ~prefix:"conv core"; *)
(*   let conv_res = layer_conv3d input conv_core ~stride:2 in *)
(*   print conv_res ~prefix:"conv_res"; *)
(*   diff conv_res; *)
(*   print input ~prefix:"input"; *)
(*   print conv_core ~prefix:"conv_core"; *)
(*   print conv_res ~prefix:"conv_res"; *)
(*   false *)
  

(* let%test "layer_conv_3d_boost"= *) 
(*   let input = random [|3;4;4|] ~bound:1. ~if_grad:false in *)
(*   let conv_core = random [|4;3;2;2|] ~bound:1. in *)
(*   for it = 0 to 3 -1  do *)
(*           (1* print some information *1) *)
(*           Printf.printf "iteration %d" it; *)
(*           print input ~prefix:"input boost"; *)
(*           print conv_core ~prefix:"conv core (boost)"; *)

(*           let conv_res = layer_conv3d_boost input conv_core ~stride:2 in *)
(*           print conv_res ~prefix:"conv_res boost"; *)

(*           diff conv_res; *)

(*           train conv_res 0.01; *)
(*           print input ~prefix:"input boost"; *)
(*           print conv_core ~prefix:"conv_core boost"; *)
(*           print conv_res ~prefix:"conv_res boost"; *)
(* done; *)
(*   false *)
