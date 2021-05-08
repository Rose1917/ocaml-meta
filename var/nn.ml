open D
type layer = {
  mutable w:var;
  mutable b:var;
  mutable acv: ?bt:meta_type -> var -> var;
}

type network = {
  layers:layer array
}
