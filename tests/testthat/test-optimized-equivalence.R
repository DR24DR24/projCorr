test_that("compiled classic components equal the pure-R reference", {
  set.seed(21)
  for (rep in 1:5) {
    x <- matrix(rnorm(36), 6, 6)
    y <- matrix(rnorm(30), 6, 5)
    perm <- sample.int(6)
    expect_equal(projCorr:::.pcov_components(x, y, perm),
                 projCorr:::.pcov_components_r(x, y, perm), tolerance = 1e-8)
  }
})

test_that("compiled IPC kernels equal the pure-R reference", {
  set.seed(22)
  for (rep in 1:5) {
    x <- matrix(rnorm(40), 8, 5)
    y <- matrix(rnorm(32), 8, 4)
    expect_equal(unclass(projCorr:::.angular_matrix(x)),
                 unclass(projCorr:::.angular_matrix_r(x)), tolerance = 1e-8)
    expect_equal(projCorr:::.ipc_quantities(x, y)[c("raw", "self_x", "self_y", "ipc2")],
                 projCorr:::.ipc_quantities_r(x, y)[c("raw", "self_x", "self_y", "ipc2")],
                 tolerance = 1e-8)
  }
})

test_that("compiled IPC permutation statistics equal direct values", {
  set.seed(23)
  x <- matrix(rnorm(50), 10, 5)
  y <- matrix(rnorm(40), 10, 4)
  values <- projCorr:::.ipc_quantities(x, y)
  perms <- rbind(sample.int(10), sample.int(10), sample.int(10))
  reference <- vapply(seq_len(nrow(perms)), function(b) {
    raw <- projCorr:::.ipcov2_from_angles_r(
      values$ax, values$ay[perms[b, ], perms[b, ], drop = FALSE]
    )
    10 * raw / values$denominator / sqrt(2)
  }, numeric(1))
  compiled <- projCorr:::cpp_ipc_permutation_stats(
    values$ax, values$ay, values$denominator, perms, 1L
  )
  expect_equal(compiled, reference, tolerance = 1e-8)
})

test_that("compiled marginal screening equals the pure-R reference", {
  set.seed(24)
  x <- matrix(rnorm(48), 12, 4)
  y <- matrix(rnorm(36), 12, 3)
  expect_equal(projCorr:::.marginal_ipc2(x, y),
               projCorr:::.marginal_ipc2_r(x, y), tolerance = 1e-8)
})

