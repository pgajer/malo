test_that("malo native spline smoother is callable", {
  x <- seq(0, 1, length.out = 25)
  y <- sin(2 * pi * x)

  fit <- malo.smooth.spline(x = x, y = y, use.gcv = FALSE, cv.repeats = 1)

  expect_true(!is.null(fit))
  expect_true(is.list(fit))
  expect_true(is.finite(fit$df))
})

test_that("C-backed legacy families are still forwarded via gflow", {
  skip_if_not_installed("gflow")

  x <- seq(0, 1, length.out = 30)
  y <- sin(2 * pi * x)

  out <- magelo(x = x, y = y, n.BB = 10, n.bws = 10, n.cv.reps = 2)
  expect_true(is.list(out))
  expect_true(is.numeric(out$gpredictions))
})
