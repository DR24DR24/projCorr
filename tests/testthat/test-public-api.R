test_that("classic PC is symmetric and invariant to location and scale", {
  set.seed(4)
  x <- matrix(rnorm(24), 12, 2)
  y <- matrix(rnorm(36), 12, 3)
  a <- pcor_test(x, y, n_perm = 7, seed = 9)
  b <- pcor_test(y, x, n_perm = 7, seed = 9)
  c <- pcor_test(2 * x + 3, -3 * y - 2, n_perm = 7, seed = 9)
  expect_equal(a$estimate, b$estimate, tolerance = 1e-10)
  expect_equal(a$estimate, c$estimate, tolerance = 1e-8)
  expect_equal(a$p.value, (1 + a$details$exceedances) / 8)
})

test_that("IPC defaults to the high-dimensional normal limit", {
  set.seed(5)
  x <- matrix(rnorm(80), 20, 4)
  y <- matrix(rnorm(60), 20, 3)
  ans <- suppressWarnings(ipcor_test(x, y))
  expect_identical(ans$details$calibration, "asymptotic")
  expect_equal(ans$p.value, pnorm(unname(ans$statistic), lower.tail = FALSE))
})

test_that("IPC auto conservatively uses corrected permutation inference", {
  set.seed(6)
  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(rnorm(30), 10, 3)
  ans <- suppressWarnings(ipcor_test(x, y, p_method = "auto", n_perm = 9, seed = 7))
  expect_identical(ans$details$calibration, "permutation")
  expect_equal(ans$p.value, (1 + ans$details$exceedances) / 10)
})

test_that("explicit seed preserves caller RNG state", {
  set.seed(11)
  before <- .Random.seed
  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(rnorm(30), 10, 3)
  before_call <- .Random.seed
  suppressWarnings(ipcor_test(x, y, p_method = "permutation", n_perm = 3, seed = 42))
  expect_identical(.Random.seed, before_call)
  expect_false(identical(before, before_call))
})

test_that("IPC is symmetric and orthogonally invariant", {
  set.seed(8)
  x <- matrix(rnorm(120), 30, 4)
  y <- matrix(rnorm(90), 30, 3)
  qx <- qr.Q(qr(matrix(rnorm(16), 4, 4)))
  qy <- qr.Q(qr(matrix(rnorm(9), 3, 3)))
  a <- suppressWarnings(ipcor_test(x, y))
  b <- suppressWarnings(ipcor_test(y, x))
  c <- suppressWarnings(ipcor_test(x %*% qx, y %*% qy))
  expect_equal(a$details$ipc_squared, b$details$ipc_squared, tolerance = 1e-10)
  expect_equal(a$details$ipc_squared, c$details$ipc_squared, tolerance = 1e-10)
})

test_that("RCV follows the paper's standardization", {
  set.seed(9)
  x <- matrix(rnorm(160), 40, 4)
  y <- matrix(rnorm(120), 40, 3)
  ans <- suppressWarnings(ipcor_rcv_test(x, y, n_pairs = 2, seed = 10))
  expect_identical(ans$details$calibration, "asymptotic")
  expect_equal(unname(ans$statistic), 40 * unname(ans$estimate) / 2)
  expect_equal(ans$p.value, pnorm(unname(ans$statistic), lower.tail = FALSE))
})

test_that("input validation catches unsupported data", {
  x <- matrix(rnorm(24), 8, 3)
  y <- matrix(rnorm(16), 8, 2)
  expect_error(ipcor_test(x, y[-1, ]), "same number")
  x[1, 1] <- NA_real_
  expect_error(ipcor_test(x, y), "Missing")
  expect_equal(ipcor_test(x, y, na.action = "omit")$details$rows_omitted, 1)
  expect_error(ipcor_rcv_test(matrix(rnorm(27), 9, 3), y), "same number")
})

test_that("heavy tails and repeated observations follow documented conventions", {
  set.seed(12)
  x <- matrix(rcauchy(120), 30, 4)
  y <- matrix(rcauchy(90), 30, 3)
  heavy <- suppressWarnings(ipcor_test(x, y))
  expect_true(all(is.finite(c(heavy$raw_estimate, heavy$statistic, heavy$p.value))))

  x[2, ] <- x[1, ]
  repeated <- suppressWarnings(ipcor_test(x, y))
  expect_true(is.finite(unname(repeated$raw_estimate)))
  expect_error(pcor_test(x, y, n_perm = 3, seed = 1), "Repeated observations")
})

test_that("a degenerate IPC denominator is reported rather than regularized", {
  x <- matrix(0, 8, 2)
  y <- matrix(rnorm(16), 8, 2)
  expect_warning(ipcor_test(x, y, sigma_x2 = 1), "non-positive")
  ans <- suppressWarnings(ipcor_test(x, y, sigma_x2 = 1))
  expect_true(is.na(ans$p.value))
  expect_true(is.na(ans$statistic))
})
