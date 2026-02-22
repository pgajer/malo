.malo.legacy.1d.warned <- new.env(parent = emptyenv())

.malo.warn.legacy.1d.api <- function(api,
                                     replacement = NULL,
                                     once = TRUE) {
    if (!is.character(api) || length(api) != 1L || !nzchar(api)) {
        stop("'api' must be a non-empty character scalar")
    }

    # In malo, these APIs are expected; warnings are opt-in to avoid double
    # warnings when gflow delegates to malo during transition.
    if (!isTRUE(getOption("malo.warn.legacy.api", FALSE))) {
        return(invisible(FALSE))
    }

    key <- paste0("warned_", api)
    if (isTRUE(once) && exists(key, envir = .malo.legacy.1d.warned, inherits = FALSE)) {
        return(invisible(FALSE))
    }
    if (isTRUE(once)) {
        assign(key, TRUE, envir = .malo.legacy.1d.warned)
    }

    msg <- paste0(
        "'", api, "' is a legacy 1D non-linear regression API now provided by malo ",
        "as part of the extraction from gflow."
    )
    if (!is.null(replacement) && nzchar(replacement)) {
        msg <- paste(msg, replacement)
    }

    warning(msg, call. = FALSE)
    invisible(TRUE)
}

# Compatibility alias used by migrated code copied from gflow.
.gflow.warn.legacy.1d.api <- .malo.warn.legacy.1d.api
