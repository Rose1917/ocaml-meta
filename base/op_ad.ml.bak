open Owl_dense_ndarray_d
(*defiantion of the general reindex operator *)
let reindex input output_shape ~map_func =
  let p = Op.Op_ad.primal input in 
        let p' = Op_base.reindex_base !p output_shape ~map_func:map_func in
  let adjf ca t = 
        let r = Op_base.reindex_reduce_base !ca (shape !p) ~map_func:map_func in
        (r,input)::t
  in
  Op.Op_ad.create p' (Op.Op_base.zeros_like p') adjf

(*defination of the general reindex_reduce operator *)
let reindex_reduce input output_shape ~map_func =
  let p = Op.Op_ad.primal input in 
        let p' = Op_base.reindex_reduce_base !p output_shape ~map_func:map_func in
  let adjf ca t = 
        let r = Op_base.reindex_base !ca (shape !p) ~map_func:map_func in
        (r,input)::t
  in
  Op.Op_ad.create p' (Op.Op_base.zeros_like p') adjf
