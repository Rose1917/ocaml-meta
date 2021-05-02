open Op.Op_ad 
let _ = 
  (*based on the auto diff module,we can easily train a network*)
  (*a network is nothing more than a function from input and various arguments to output*)

  (*the neural network defination*)
  let f w1 b1 w2 b2 input target=
      sum (sqr (sub (sigmoid (add b2 (mat_mul w2 (sigmoid (add (mat_mul w1 input) b1))))) target)) in

  (*prepare for the neural network arguments,since we do not need the d(loss)/d(input),so we set the if_grad = false*)
  (*for convinence,we set the arguments sequentially*)
  let input =  (sequential ~if_grad:false [|2;1|]) in
  let w1 = sequential [|2;2|] in
  let b1 = sequential [|2;1|] in
  let w2 = sequential [|2;2|] in
  let b2 = sequential [|2;1|] in
  let target = sequential [|2;1|] in

  (*iterate to train,every diff function will do a forward propogation and backward propogation*)
  (*train function will do a update according to the gradient set before by diff function*)

  let step = 0.01 in
  let cycles = 1000 in
  Printf.printf "iteration begin,cycles %d ,step %f\n" cycles step;
  Stdlib.print_endline "iteration begin";
  for it = 1 to cycles do
    
    let z = diff f w1 b1 w2 b2 input target in
    (*
    print input;
    print w1;
    print b1;
    print w2;
    print b2;
    print z;
    *)

    train z step;

    (*
    print input;
    print w1;
    print b1;
    print w2;
    print b2;
    print z;
    *)
    let loss = get_ele z [|0|] in
    Printf.printf "iteration %d,loss %f\n" it loss 
  done 
