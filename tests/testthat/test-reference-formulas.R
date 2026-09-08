test_that("IPC reduction equals direct distinct-index U-statistic", {
  set.seed(1)
  for (n in 4:7) {
    a <- projCorr:::.angular_matrix_r(matrix(rnorm(n * 2), n, 2))
    b <- projCorr:::.angular_matrix_r(matrix(rnorm(n * 3), n, 3))
    expect_equal(projCorr:::.ipcov2_from_angles_r(a, b),
                 projCorr:::.ipcov2_direct_r(a, b), tolerance = 1e-10)
  }
})

test_that("classic centered formula equals explicit V-statistic expansion", {
  set.seed(2)
  x <- matrix(rnorm(12), 6, 2)
  y <- matrix(rnorm(18), 6, 3)
  got <- projCorr:::.pcov_components_r(x, y)$pcov2
  direct <- 0
  for (r in 1:6) {
    a <- projCorr:::.anchor_angle_r(x, r)
    b <- projCorr:::.anchor_angle_r(y, r)
    direct <- direct + sum(projCorr:::.double_center_r(a) *
                             projCorr:::.double_center_r(b))
  }
  expect_equal(got, direct / 6^3, tolerance = 1e-12)
})

test_that("angular kernel has paper scale and diagonal conventions", {
  z <- rbind(c(1, 0), c(0, 1), c(1, 1))
  a <- projCorr:::.angular_matrix_r(z, sigma2 = 2)
  expected <- acos((2 + sum(z[1, ] * z[2, ])) /
                     sqrt((2 + sum(z[1, ]^2)) * (2 + sum(z[2, ]^2))))
  expect_equal(a[1, 2], expected)
  expect_equal(diag(a), rep(0, 3))
})

