(* define the boost type. by default it will not boost*)
open Util.Mnist
open Op.Op_base
let _ = 
  let _test_img,_test_label,_test_label_h = load_test_data() in
  for it = 0 to 10000 - 1 do
        let img_test = reshape (Bigarray.Genarray.slice_left _test_img [|it|]) [|28;28|] in
        draw_image img_test;
  done
  (*print train_set ~prefix:"train:set";;*)
