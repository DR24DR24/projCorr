.as_numeric_matrix <- function(x, name) {
  if (is.data.frame(x)) {
    if (!all(vapply(x, is.numeric, logical(1)))) {
      stop(sprintf("`%s` must contain only numeric columns.", name), call. = FALSE)
    }
    x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || ncol(x) < 1L) {
    stop(sprintf("`%s` must be a numeric matrix or numeric data frame with at least one column.",
                 name), call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

.validate_paired_data <- function(x, y, na.action, min_n) {
  x <- .as_numeric_matrix(x, "x")
  y <- .as_numeric_matrix(y, "y")
  if (nrow(x) != nrow(y)) {
    stop("`x` and `y` must have the same number of rows.", call. = FALSE)
  }
  na.action <- match.arg(na.action, c("fail", "omit"))
  finite <- apply(x, 1L, function(z) all(is.finite(z))) &
    apply(y, 1L, function(z) all(is.finite(z)))
  omitted <- sum(!finite)
  if (omitted && na.action == "fail") {
    stop("Missing or non-finite values are not allowed when `na.action = \"fail\"`.",
         call. = FALSE)
  }
  if (omitted) {
    x <- x[finite, , drop = FALSE]
    y <- y[finite, , drop = FALSE]
  }
  if (nrow(x) < min_n) {
    stop(sprintf("At least %d complete paired observations are required.", min_n),
         call. = FALSE)
  }
  list(x = x, y = y, omitted = omitted)
}

.validate_count <- function(x, name, minimum = 1L) {
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x != floor(x) ||
      x < minimum || x > .Machine$integer.max) {
    stop(sprintf("`%s` must be a single integer of at least %d.", name, minimum),
         call. = FALSE)
  }
  as.integer(x)
}

.validate_seed <- function(seed) {
  if (is.null(seed)) return(NULL)
  if (length(seed) != 1L || is.na(seed) || !is.finite(seed) ||
      seed != floor(seed) || seed < 0 || seed > .Machine$integer.max) {
    stop("`seed` must be NULL or a non-negative integer.", call. = FALSE)
  }
  as.integer(seed)
}

.with_local_seed <- function(seed, code) {
  if (is.null(seed)) return(force(code))
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  force(code)
}

.permutation_matrix <- function(n, n_perm) {
  identity <- seq_len(n)
  out <- matrix(0L, nrow = n_perm, ncol = n)
  for (b in seq_len(n_perm)) {
    repeat {
      candidate <- sample.int(n)
      if (!identical(candidate, identity)) break
    }
    out[b, ] <- candidate
  }
  out
}

.make_htest <- function(estimate, raw_estimate, statistic, p.value,
                        parameter, method, data.name, details,
                        null_name = "correlation") {
  out <- list(
    statistic = statistic,
    parameter = parameter,
    p.value = p.value,
    estimate = estimate,
    raw_estimate = raw_estimate,
    null.value = stats::setNames(0, null_name),
    alternative = "greater",
    method = method,
    data.name = data.name,
    details = details
  )
  class(out) <- "htest"
  out
}

.safe_sqrt_estimate <- function(value, label, tol = 1e-12) {
  if (!is.finite(value)) return(NA_real_)
  if (value < -tol) {
    warning(sprintf("The finite-sample %s is negative; its square-root estimate is undefined.",
                    label), call. = FALSE)
    return(NA_real_)
  }
  sqrt(max(value, 0))
}

