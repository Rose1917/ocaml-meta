open Var.D
type layer = {
  mutable w:var;
  mutable b:var;
  a: var -> var;
}

type network = {
  layers:layer array
}
