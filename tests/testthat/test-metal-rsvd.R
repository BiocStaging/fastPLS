test_that("standalone Metal rSVD rejects operation-split execution", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(42)
  A <- matrix(rnorm(80 * 25), 80, 25)
  expect_error(
    fastPLS::fastsvd(
      A,
      ncomp = 5,
      backend = "metal",
      oversample = 8,
      power = 1,
      seed = 123
    ),
    "fully device-native"
  )
})
