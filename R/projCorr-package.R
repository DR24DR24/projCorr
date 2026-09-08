#' projCorr: Projection Correlation Independence Tests
#'
#' Implements classic and improved projection-correlation independence tests.
#' Improved IPC and IPC-RCV default to their high-dimensional normal limits;
#' use `p_method = "permutation"` for finite-sample randomization calibration.
#'
#' @useDynLib projCorr, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @keywords internal
"_PACKAGE"
