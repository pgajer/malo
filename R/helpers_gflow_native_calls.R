.malo.C <- function(name, ..., PACKAGE = "malo") {
  expr <- substitute(name)
  nm <- if (is.character(expr)) expr else as.character(expr)
  .C(nm, ..., PACKAGE = PACKAGE)
}

.malo.Call <- function(name, ..., PACKAGE = "malo") {
  expr <- substitute(name)
  nm <- if (is.character(expr)) expr else as.character(expr)
  .Call(nm, ..., PACKAGE = PACKAGE)
}
