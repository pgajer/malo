# Weighted Pearson helpers required by migrated magelo routines.
pearson.wcor <- function(x, y, w) {
    if (!is.numeric(x) || !is.numeric(y) || !is.numeric(w)) {
        stop("x, y, and w must be numeric vectors")
    }

    n <- length(x)

    if (length(y) != n || length(w) != n) {
        stop("x, y, and w must have the same length")
    }

    if (any(w < 0)) {
        stop("weights must be non-negative")
    }

    cc <- 0
    out <- .malo.C("C_pearson_wcor",
                   as.double(x),
                   as.double(y),
                   as.double(w),
                   as.integer(n),
                   cc = as.double(cc))
    out$cc
}

pearson.wcor.BB.qCrI <- function(nn.y1, nn.y2, nn.i, nn.w, nx, n.BB = 1000, alpha = 0.05) {
    if (!is.matrix(nn.y1) || !is.matrix(nn.y2) || !is.matrix(nn.i) || !is.matrix(nn.w)) {
        stop("nn.y1, nn.y2, nn.i, and nn.w must be matrices")
    }

    K <- ncol(nn.w)
    ng <- nrow(nn.w)

    if (ncol(nn.y1) != K || nrow(nn.y1) != ng ||
        ncol(nn.y2) != K || nrow(nn.y2) != ng ||
        ncol(nn.i) != K || nrow(nn.i) != ng) {
        stop("All input matrices must have the same dimensions")
    }

    if (n.BB < 1) {
        stop("n.BB must be at least 1")
    }

    if (alpha <= 0 || alpha >= 1) {
        stop("alpha must be between 0 and 1")
    }

    lwcor.CI <- numeric(2 * ng)
    out <- .malo.C("C_pearson_wcor_BB_qCrI",
                   as.double(t(nn.y1)),
                   as.double(t(nn.y2)),
                   as.integer(t(nn.i - 1)),
                   as.double(t(nn.w)),
                   as.integer(K),
                   as.integer(ng),
                   as.integer(nx),
                   as.integer(n.BB),
                   as.double(alpha),
                   lwcor.CI = as.double(lwcor.CI))

    matrix(out$lwcor.CI, nrow = 2, ncol = ng, byrow = FALSE)
}
