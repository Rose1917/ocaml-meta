(* define the boost type. by default it will not boost*)
open Base.Op_base
let _ = 
  set_boost FMA_BOOST;
  for _ = 1 to 100 do
  let x = sequential [|2;3|] in
  let y = sequential [|3;2|] in
  let z = mat_dot x y in
  print x;
  print y;
  print z;
done;;
