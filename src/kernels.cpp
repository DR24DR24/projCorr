// Computational kernels adapted from the local MIT-licensed projIndep package.
// See inst/COPYRIGHTS for attribution.
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct PcovComponents {
  double pcov2;
  double s2;
  bool valid;
};

arma::mat anchor_angles(const arma::mat& z, const arma::uvec& map,
                        arma::uword anchor, bool& valid) {
  const arma::uword n = z.n_rows;
  arma::mat unit(n, z.n_cols, arma::fill::zeros);
  const arma::rowvec origin = z.row(map[anchor]);
  for (arma::uword i = 0; i < n; ++i) {
    if (i == anchor) continue;
    arma::rowvec delta = z.row(map[i]) - origin;
    const double norm = arma::norm(delta, 2);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      valid = false;
      return arma::mat();
    }
    unit.row(i) = delta / norm;
  }
  arma::mat cosine = unit * unit.t();
  cosine = arma::clamp(cosine, -1.0, 1.0);
  arma::mat angle = arma::acos(cosine);
  angle.row(anchor).zeros();
  angle.col(anchor).zeros();
  return angle;
}

arma::mat double_center(const arma::mat& x) {
  arma::mat out = x;
  out.each_col() -= arma::mean(x, 1);
  out.each_row() -= arma::mean(x, 0);
  out += arma::accu(x) / static_cast<double>(x.n_elem);
  return out;
}

PcovComponents pcov_components_impl(const arma::mat& x, const arma::mat& y,
                                    const arma::uvec& map_y) {
  const arma::uword n = x.n_rows;
  arma::uvec identity = arma::regspace<arma::uvec>(0, n - 1);
  double cross_sum = 0.0;
  double s2_sum = 0.0;
  bool valid = true;
  for (arma::uword anchor = 0; anchor < n; ++anchor) {
    arma::mat ax = anchor_angles(x, identity, anchor, valid);
    if (!valid) return {NA_REAL, NA_REAL, false};
    arma::mat ay = anchor_angles(y, map_y, anchor, valid);
    if (!valid) return {NA_REAL, NA_REAL, false};
    cross_sum += arma::accu(double_center(ax) % double_center(ay));
    s2_sum += arma::accu(ax) * arma::accu(ay);
  }
  const double nd = static_cast<double>(n);
  return {cross_sum / (nd * nd * nd),
          s2_sum / (nd * nd * nd * nd * nd), true};
}

double ipcov2_impl(const arma::mat& a, const arma::mat& b) {
  const double n = static_cast<double>(a.n_rows);
  const double sum_ab = arma::accu(a % b);
  const double row_product = arma::dot(arma::sum(a, 1), arma::sum(b, 1));
  const double total_product = arma::accu(a) * arma::accu(b);
  return sum_ab / (n * (n - 1.0)) -
    2.0 * (row_product - sum_ab) / (n * (n - 1.0) * (n - 2.0)) +
    (total_product - 4.0 * row_product + 2.0 * sum_ab) /
      (n * (n - 1.0) * (n - 2.0) * (n - 3.0));
}

double median_copy(arma::vec x) {
  x = arma::sort(x);
  const arma::uword n = x.n_elem;
  if (n % 2 == 1) return x[n / 2];
  return 0.5 * (x[n / 2 - 1] + x[n / 2]);
}

arma::mat angular_matrix_impl(const arma::mat& z, double sigma2, bool& valid) {
  const arma::vec norm2 = arma::sum(arma::square(z), 1) + sigma2;
  if (!std::isfinite(sigma2) || sigma2 < 0.0 || arma::any(norm2 <= 0.0) ||
      !norm2.is_finite()) {
    valid = false;
    return arma::mat();
  }
  arma::mat cosine = z * z.t();
  cosine += sigma2;
  cosine.each_col() /= arma::sqrt(norm2);
  cosine.each_row() /= arma::sqrt(norm2).t();
  cosine = arma::clamp(cosine, -1.0, 1.0);
  arma::mat out = arma::acos(cosine);
  out.diag().zeros();
  valid = true;
  return out;
}

arma::mat feature_angles(const arma::vec& z, bool& valid) {
  const double sigma2 = median_copy(arma::square(z));
  arma::mat column(z);
  return angular_matrix_impl(column, sigma2, valid);
}

arma::uvec checked_permutation(const Rcpp::IntegerVector& perm, arma::uword n) {
  if (static_cast<arma::uword>(perm.size()) != n) {
    Rcpp::stop("Permutation length does not match the sample size.");
  }
  arma::uvec out(n);
  std::vector<bool> seen(n, false);
  for (arma::uword i = 0; i < n; ++i) {
    const int value = perm[i] - 1;
    if (value < 0 || static_cast<arma::uword>(value) >= n || seen[value]) {
      Rcpp::stop("Invalid permutation indices.");
    }
    out[i] = static_cast<arma::uword>(value);
    seen[value] = true;
  }
  return out;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_pcor_components(const arma::mat& x, const arma::mat& y,
                               Rcpp::IntegerVector perm) {
  if (x.n_rows != y.n_rows || x.n_rows < 2) {
    Rcpp::stop("Incompatible sample sizes for projection covariance.");
  }
  arma::uvec map = checked_permutation(perm, y.n_rows);
  PcovComponents value = pcov_components_impl(x, y, map);
  if (!value.valid) {
    Rcpp::stop("Repeated observations make a projection angle undefined.");
  }
  return Rcpp::List::create(
    Rcpp::Named("pcov2") = value.pcov2,
    Rcpp::Named("s2") = value.s2
  );
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_pcor_permutation_stats(
    const arma::mat& x, const arma::mat& y, const Rcpp::IntegerMatrix& perms,
    int n_threads) {
  const int n_perm = perms.nrow();
  const arma::uword n = x.n_rows;
  std::vector<double> out(n_perm, NA_REAL);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int b = 0; b < n_perm; ++b) {
    arma::uvec map(n);
    bool valid_map = true;
    std::vector<bool> seen(n, false);
    for (arma::uword i = 0; i < n; ++i) {
      const int value = perms(b, i) - 1;
      if (value < 0 || static_cast<arma::uword>(value) >= n || seen[value]) {
        valid_map = false;
        break;
      }
      map[i] = static_cast<arma::uword>(value);
      seen[value] = true;
    }
    if (!valid_map) continue;
    PcovComponents value = pcov_components_impl(x, y, map);
    const double denominator = arma::datum::pi * arma::datum::pi - value.s2;
    if (value.valid && denominator > 0.0) {
      out[b] = static_cast<double>(n) * value.pcov2 / denominator;
    }
  }
  return Rcpp::wrap(out);
}

// [[Rcpp::export]]
arma::mat cpp_angular_matrix(const arma::mat& z, double sigma2) {
  bool valid = true;
  arma::mat out = angular_matrix_impl(z, sigma2, valid);
  if (!valid) {
    Rcpp::stop("The scale and row norms yield an undefined angular kernel.");
  }
  return out;
}

// [[Rcpp::export]]
double cpp_ipcov2_from_angles(const arma::mat& a, const arma::mat& b) {
  if (a.n_rows != a.n_cols || b.n_rows != b.n_cols ||
      a.n_rows != b.n_rows || a.n_rows < 4) {
    Rcpp::stop("Angular matrices must be square, equally sized, and at least 4 by 4.");
  }
  return ipcov2_impl(a, b);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_ipc_permutation_stats(
    const arma::mat& a, const arma::mat& b, double denominator,
    const Rcpp::IntegerMatrix& perms, int n_threads) {
  const int n_perm = perms.nrow();
  const arma::uword n = a.n_rows;
  const arma::vec row_a = arma::sum(a, 1);
  const arma::vec row_b = arma::sum(b, 1);
  const double total_product = arma::accu(a) * arma::accu(b);
  const double nd = static_cast<double>(n);
  std::vector<double> out(n_perm, NA_REAL);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int rep = 0; rep < n_perm; ++rep) {
    arma::uvec map(n);
    bool valid = true;
    std::vector<bool> seen(n, false);
    for (arma::uword i = 0; i < n; ++i) {
      const int value = perms(rep, i) - 1;
      if (value < 0 || static_cast<arma::uword>(value) >= n || seen[value]) {
        valid = false;
        break;
      }
      map[i] = static_cast<arma::uword>(value);
      seen[value] = true;
    }
    if (!valid) continue;
    double sum_ab = 0.0;
    double row_product = 0.0;
    for (arma::uword i = 0; i < n; ++i) {
      row_product += row_a[i] * row_b[map[i]];
      for (arma::uword j = 0; j < n; ++j) {
        sum_ab += a(i, j) * b(map[i], map[j]);
      }
    }
    const double raw = sum_ab / (nd * (nd - 1.0)) -
      2.0 * (row_product - sum_ab) / (nd * (nd - 1.0) * (nd - 2.0)) +
      (total_product - 4.0 * row_product + 2.0 * sum_ab) /
        (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0));
    out[rep] = nd * (raw / denominator) / std::sqrt(2.0);
  }
  return Rcpp::wrap(out);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_marginal_ipc2(const arma::mat& x, const arma::mat& y) {
  const arma::uword p = x.n_cols;
  const arma::uword q = y.n_cols;
  Rcpp::NumericMatrix out(p, q);
  std::fill(out.begin(), out.end(), NA_REAL);

  const bool store_x = p <= q;
  const arma::mat& small = store_x ? x : y;
  const arma::mat& large = store_x ? y : x;
  std::vector<arma::mat> angles(small.n_cols);
  std::vector<double> self(small.n_cols, NA_REAL);
  std::vector<bool> usable(small.n_cols, false);

  for (arma::uword i = 0; i < small.n_cols; ++i) {
    bool valid = true;
    angles[i] = feature_angles(small.col(i), valid);
    if (valid) {
      self[i] = ipcov2_impl(angles[i], angles[i]);
      usable[i] = std::isfinite(self[i]) && self[i] > 0.0;
    }
  }

  for (arma::uword j = 0; j < large.n_cols; ++j) {
    Rcpp::checkUserInterrupt();
    bool valid = true;
    arma::mat other = feature_angles(large.col(j), valid);
    if (!valid) continue;
    const double other_self = ipcov2_impl(other, other);
    if (!(std::isfinite(other_self) && other_self > 0.0)) continue;
    for (arma::uword i = 0; i < small.n_cols; ++i) {
      if (!usable[i]) continue;
      const double value = ipcov2_impl(angles[i], other) /
        std::sqrt(self[i] * other_self);
      if (store_x) out(i, j) = value;
      else out(j, i) = value;
    }
  }
  return out;
}

// [[Rcpp::export]]
bool cpp_openmp_available() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
