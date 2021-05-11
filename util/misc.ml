let first tuple = 
  match tuple with
  |(x,_) -> x
let second tuple = 
  match tuple with 
  |(_,y) -> y

let max_i x =
  let l = Array.length x in
  let max_index = ref (-1) in
  let max_ele   = ref (Float.min_float) in
  for it = 0 to l - 1 do
    if (x.(it) > !max_ele) then (max_ele := x.(it);max_index := it)
  done;
  !max_index

let max x =
  let l = Array.length x in
  let max_index = ref (-1) in
  let max_ele   = ref (Float.min_float) in
  for it = 0 to l - 1 do
    if (x.(it) > !max_ele) then( max_ele := x.(it);max_index := it)
  done;
  !max_ele

let min_i x =
  let l = Array.length x in
  let min_index = ref (-1) in
  let min_ele   = ref (Float.max_float) in
  for it = 0 to l - 1 do
    if (x.(it) < !min_ele) then (min_ele := x.(it);min_index := it)
  done;
  !min_index

let min x =
  let l = Array.length x in
  let min_index = ref (-1) in
  let min_ele   = ref (Float.max_float) in
  for it = 0 to l - 1 do
    if (x.(it) < !min_ele) then( min_ele := x.(it);min_index := it)
  done;
  !min_ele

let non_zeroi x = 
  let l = Array.length x in
  let res = ref (l - 1) in
  for it = l -1  downto 0 do
    if x.(it) <> 0. then res:= it
  done;
  !res
let%test _=non_zeroi [|2.;3.;4.|] = 0
let%test _=non_zeroi [|0.;3.;4.|] = 1

let non_zero x = 
  let l = Array.length x in
  let res = ref (0.) in
  for it = l -1  downto 0 do
    if x.(it) <> 0. then res:= x.(it)
  done;
  !res

