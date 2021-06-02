(* open Op.Op_base *) 
(* (1* define the boost type. by default it will not boost *1) *)
(* let%test "dot" = *)  
(*   let cycles        = 1000 in *) 
(*   let x             = sequential [|10;98|] in *) 
(*   let y             = sequential [|98;1|] in *) 
(*   let z             = ref (sequential [|10;1|]) in *) 
(*   for _ = 0 to cycles - 1 do *) 
(*     z := dot x y *)  
(*   done; *) 

(*   print x ; *) 
(*   print y ; *) 
(*   print !z ; *) 
(*   true *) 
open Op.Op_base
let %test "cuda_mat" =
        let x = sequential [|10;10|] in
        let y = sequential [|10;10|] in
        let z = cuda_mat_mul x y in
        print x;
        print y;
        print z;
        false
