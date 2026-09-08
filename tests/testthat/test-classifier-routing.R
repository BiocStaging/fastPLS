test_that("public classifier names route through backend", {
  expect_equal(fastPLS:::.resolve_classifier_for_backend("argmax", "cpu"), "argmax")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "cpu"), "lda_cpp")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "cuda"), "lda_cuda")
  expect_equal(fastPLS:::.resolve_classifier_for_backend("lda", "metal"), "lda_cpp")
  expect_error(fastPLS:::.resolve_classifier_for_backend("cknn", "cpu"))
  expect_error(fastPLS:::.resolve_classifier_for_backend("candidate_knn", "cpu"))
  expect_error(fastPLS:::.normalize_classifier_public("lda_cuda"))
})

test_that("retired classifier ABI is absent and top-k argmax remains available", {
  ns <- asNamespace("fastPLS")
  expect_false(exists("candidate_knn_predict_cpp", ns, inherits = FALSE))
  expect_false(exists("candidate_knn_predict_cuda", ns, inherits = FALSE))
  expect_false(exists(".class_bias_predict", ns, inherits = FALSE))

  fit <- pls(
    as.matrix(iris[, 1:4]),
    iris$Species,
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    classifier = "argmax",
    return_variance = FALSE
  )
  pred <- predict(fit, as.matrix(iris[1:8, 1:4]), top = 3L)
  expect_identical(dim(pred$Ypred_top[[1L]]), c(8L, 3L))
  expect_identical(dim(pred$Ypred_top_score[[1L]]), c(8L, 3L))
})
