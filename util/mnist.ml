
(*the mnist file path*)
let train_image_path = "dataset/train-images-idx3-ubyte"
let train_label_path = "dataset/train-labels-idx1-ubyte"
let test_image_path  = "dataset/t10k-images-idx3-ubyte"
let test_label_path  = "dataset/t10k-labels-idx1-ubyte"

(*some global variable*)
let train_epoc = 60000
let test_epoc  = 10000
let pixel_num  = 28*28

(*to load the train iamge and label*)
let load_train_images () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|60000;784|] in
  let oi  = open_in_bin train_image_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    let rows      = input_binary_int oi in
    let cols      = input_binary_int oi in 
    Printf.printf "Reading the train images:magic number:%d image number:%d rows: %d columns %d\n" magic_num image_num rows cols;
    for it_x = 0 to 60000 - 1 do
        for it_y = 0 to 784 - 1 do 
                let pixel = input_byte oi in
                Bigarray.Genarray.set res [|it_x;it_y|] ((Float.of_int pixel)/.256.);
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "train images reading error" 
    
let load_train_labels () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|60000;1|] in
  let oi  = open_in_bin train_label_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    Printf.printf "Reading the train labels:magic number:%d image number:%d\n" magic_num image_num;
    for it_x = 0 to 60000 - 1 do
        for it_y = 0 to 1 - 1 do 
                let pixel = input_byte oi in
                Bigarray.Genarray.set res [|it_x;it_y|] (Float.of_int pixel);
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "train labels reading error" 

let load_train_labels_hot () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|60000;10|] in
  let oi  = open_in_bin train_label_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    Printf.printf "Reading the train labels one hot:magic number:%d image number:%d\n" magic_num image_num ;
    for it_x = 0 to 60000 - 1 do
        let pixel = input_byte oi in
        for it_y = 0 to 10 - 1 do 
          if (it_y = pixel) then Bigarray.Genarray.set res [|it_x;it_y|] 1.
          else Bigarray.Genarray.set res [|it_x;it_y|] 0.
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "train labels reading error" 


let load_train_data () =
  (load_train_images(),load_train_labels(),load_train_labels_hot())

(*to load the test iamge and label*)
let load_test_images () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|10000;784|] in
  let oi  = open_in_bin test_image_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    let rows      = input_binary_int oi in
    let cols      = input_binary_int oi in 
    Printf.printf "Reading the test images:magic number:%d image number:%d rows: %d columns %d\n" magic_num image_num rows cols;
    for it_x = 0 to 10000 - 1 do
        for it_y = 0 to 784 - 1 do 
                let pixel = input_byte oi in
                Bigarray.Genarray.set res [|it_x;it_y|] ((Float.of_int pixel)/.256.);
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "test images reading error" 
    
let load_test_labels () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|10000;1|] in
  let oi  = open_in_bin test_label_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    Printf.printf "Reading the test labels:magic number:%d image number:%d\n" magic_num image_num;
    for it_x = 0 to 10000 - 1 do
        for it_y = 0 to 1 - 1 do 
                let pixel = input_byte oi in
                Bigarray.Genarray.set res [|it_x;it_y|] (Float.of_int pixel);
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "test labels reading error" 

let load_test_labels_hot () = 
  let res = Bigarray.Genarray.create (Bigarray.float64) (Bigarray.c_layout) [|10000;10|] in
  let oi  = open_in_bin test_label_path in
  try 
    let magic_num = input_binary_int oi in
    let image_num = input_binary_int oi in
    Printf.printf "Reading the test labels one hot:magic number:%d image number:%d\n" magic_num image_num ;
    for it_x = 0 to 10000 - 1 do
        let pixel = input_byte oi in
        for it_y = 0 to 10 - 1 do 
          if (it_y = pixel) then Bigarray.Genarray.set res [|it_x;it_y|] 1.
          else Bigarray.Genarray.set res [|it_x;it_y|] 0.
        done
    done;
    close_in oi;
    res
  with _ -> 
    failwith "test labels (one hot) reading error" 


let load_test_data () =
  (load_test_images(),load_test_labels(),load_test_labels_hot())
(*draw the image *)
let draw_image x = 
  let x = Bigarray.reshape x [|28;28|] in
  let s = Bigarray.Genarray.dims x in
  let r = s.(0) in
  let c = s.(1) in
  for it_x = 0 to r - 1 do
    for it_y = 0 to c - 1 do
      if (Bigarray.Genarray.get x [|it_x;it_y|] = 0. ) 
         then Printf.printf " " 
      else 
          Printf.printf "*"
     done;
   Printf.printf "\n"
  done;
 Printf.printf "\n"

