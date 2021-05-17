open Op.Op_base
(* define the boost type. by default it will not boost*)
let%test _ = 
  let cycles        = 1000 in
  let x             = sequential [|10;10|] in
  let y             = sequential [|10;10|] in
  let z             = ref (sequential [|10;10|]) in
  for _ = 0 to cycles - 1 do
    z := dot x y 
  done;

  print x ;
  print y ;
  print !z ;
  true
