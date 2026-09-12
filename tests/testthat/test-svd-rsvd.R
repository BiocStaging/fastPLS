test_that("cpu_rsvd approximates exact SVD on dense matrices", {
  set.seed(1001)
  A <- matrix(rnorm(80 * 30), nrow = 80, ncol = 30)

  exact <- base::svd(A, nu = 6L, nv = 6L)
  rsvd <- suppressWarnings(fastsvd(
    A, ncomp = 6L, backend = "cpu",
    oversample = 12L, power = 2L, seed = 7L
  ))

  expect_equal(length(exact$d[1:6]), 6L)
  expect_equal(length(rsvd$d), 6L)
  expect_equal(rsvd$d, exact$d[1:6], tolerance = 1e-2)

  proj_exact <- exact$u[, 1:6, drop = FALSE] %*%
    t(exact$u[, 1:6, drop = FALSE])
  proj_rsvd <- rsvd$u %*% t(rsvd$u)
  expect_lt(norm(proj_exact - proj_rsvd, type = "F"), 0.5)
})
