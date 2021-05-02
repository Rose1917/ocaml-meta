type base_t = Owl_dense_ndarray_d.arr
type var = {
  mutable base_val : base_t ref;
  mutable deri_val : base_t ref; 
  mutable adj_fun : base_t ref -> (base_t * var) list -> (base_t * var) list;
  mutable if_trained:bool ref;
  mutable if_grad:bool ref;
}
