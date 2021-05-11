(* define the boost type. by default it will not boost*)
open Nn.Full
open Op.Op_ad
open Util.Mnist

let if_match x y = 
  Printf.printf "=======\n";
  let x_i = x|>primal|>(!)|>Op.Op_base.to_arr|>Util.Misc.max_i in
  let y_i = y|>primal|>(!)|>Op.Op_base.to_arr|>Util.Misc.max_i in
  Printf.printf "if matched:the x max i is %d, y is %d\n" x_i y_i;
  for i = 0 to 9 do
    Printf.printf "pred value %d :%f\n" i ((y|>primal|>(!)|>Op.Op_base.to_arr).(i))
  done;
  if x_i = y_i then 1 else 0

let _ = 
  (*set the boost : OMP_BOOST FMA_BOOST AVX_BOOST*)
  set_boost OMP_BOOST;  
  let layer_1 = ref (liner 10 (28*28) sigmoid) in
  let layer_2 = ref (liner 10 10 softmax) in
  let test_network = init_net [|!layer_1;!layer_2;|] in

  (*the neural network defination*)

  (*prepare for the neural network arguments,since we do not need the d(loss)/d(input),so we set the if_grad = false*)
  (*for convinence,we set the arguments sequentially*)

  (*iterate to train,every diff function will do a forward propogation and backward propogation*)
  (*train function will do a update according to the gradient set before by diff function*)

  let train_set,_,train_label = load_train_data() in 
  let step = 0.05 in
  let cycles = train_epoc in
  for it = 0 to 10000 - 1 do
    let input  = pack (reshape (slice train_set [|(it,1);(0,784)|]) [|(28*28);1|])~if_grad:false in
    let target = pack (reshape (slice train_label [|(it,1);(0,10)|]) [|10;1|]) in
    let output =  run_net input test_network in
    let loss   = cross_entry output target in
    let loss_float = get_ele loss [|0|] in
    Printf.printf "iteration %d,loss %g(before trained)\n" it loss_float;
    diff loss;
    train loss step;
    draw_image (input |> unpack |> (!));
    print output  ~prefix:"the prediction value of neural network";
  done; 
  Printf.printf "train iteration begin,cycles %d ,step %f\n" cycles step;
  Stdlib.print_endline "iteration begin";

  (*
  for it = 0 to 1 - 1 do
    Printf.printf "=======\n";

    let input = pack (reshape (slice train_set [|(it,1);(0,784)|]) [|(28*28);1|])  ~if_grad:false in
    let target = pack (reshape (slice train_label [|(it,1);(0,10)|]) [|10;1|])  in 
    let z =  run_net input test_network in
    let loss = cross_entry z target in
    Util.Mnist.draw_image (input|>unpack|> (!) );
    diff loss;
    print z ~prefix:"the output of neural network";
    (*
    print loss ~prefix:"loss";
    print z ~prefix:"z";
    print !layer_2.w ~prefix:"layer_2 w";
    print !layer_2.b ~prefix:"layer_2 b";
       *)
    let loss_float = get_ele loss [|0|] in
    Printf.printf "iteration %d,loss %g(before trained)\n" it loss_float;
    train loss step;
    print !layer_2.b ~prefix:"layer 2 bias after trained";
    let z = run_net input test_network in
    let loss = cross_entry z target in
    print z ~prefix:"z after trained";
    print loss ~prefix:"loss(after trained)";
    (*
    let z =run_net input test_network in
    print z ~prefix:" z after trained output";
       *)
  done;

  for it = 0 to 1 - 1 do
    let input = pack (reshape (slice train_set [|(it,1);(0,784)|]) [|(28*28);1|]) ~if_grad:false in
    let target = pack (reshape (slice train_label [|(it,1);(0,10)|]) [|10;1|])  in 
    let z =  run_net input test_network in
    draw_image (input |> unpack |> (!));
    print target  ~prefix:"label value of nn";
    print z  ~prefix:"the prediction value of neural network";
  done; 
  (*
  let match_cnt = ref 0 in
  let cycles = test_epoc in
  let test_set,_,test_label = load_test_data () in
  Printf.printf "test iteration begin,cycles %d\n" cycles;
  Stdlib.print_endline "iteration begin";
  for it = 0 to 10000 - 1 do
    let input = pack (reshape (slice test_set [|(it,1);(0,784)|]) [|(28*28);1|]) ~if_grad:false in
    let target = pack (reshape (slice test_label [|(it,1);(0,10)|]) [|10;1|]) ~if_grad:false in 
    let z =  run_net input test_network in
    draw_image (input |> unpack |> (!));
    match_cnt:= !match_cnt + (if_match target z);
  done; 
  let _a = Printf.printf "%d matched the accuray is %g\n" !match_cnt ((Float.of_int !match_cnt)/. 10000.) in
  ()
     *)
     *)
