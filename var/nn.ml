open D
type full_layer = {
  mutable w:var;
  mutable b:var;
  mutable act: ?bt:meta_type -> var -> var;
}

type conv_layer = {
  mutable w:var;
  mutable b:var;
  mutable stride:int;
  mutable pad:int;
  mutable act: ?bt:meta_type -> var -> var;
}

type pool_type =
  |MaxPool
  |MeanPool

type pool_layer = {
  mutable h:int;
  mutable w:int;
  mutable stride:int;
  mutable t:pool_type;
}

type layer = 
  |Full of full_layer
  |Conv of conv_layer
  |Pool of pool_layer
  |Connect

type network = {
  layers:layer array
}
