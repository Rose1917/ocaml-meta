open Op.Op_base
(* define the boost type. by default it will not boost*)
let bt_of_string s = 
  match s with
  |"OMP" -> OMP_BOOST
  |"AVX" -> AVX_BOOST
  |"FMA" -> FMA_BOOST
  |_   -> DEFAULT

let _ =
  let bt            = Sys.argv.(2) |> bt_of_string in
  set_boost bt;
  let cycles        = int_of_string (Sys.argv.(1))  in
  Printf.printf "the cycles %i \n" cycles;
  let x             = ref (random [|10;10|]) in
  let y             = ref (random [|10;10|]) in
  let z             = ref (random [|10;10|]) in
  for _ = 0 to cycles - 1 do
    x      := random [|10;10|];
    y      := random [|10;10|];
    z := mat_dot !x !y ;
      done;

  print !x ;
  print !y ;
  print !z 
