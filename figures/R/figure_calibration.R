# Supplementary figure: model against observed cases, admissions and deaths.
#   Rscript figures/R/figure_calibration.R <data dir> <output dir>
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
args    <- commandArgs(trailingOnly = TRUE)
in_dir  <- if (length(args) >= 1) args[1] else "paper_figures"
out_dir <- if (length(args) >= 2) args[2] else "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
here <- function() {
  d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  if (is.na(d) || !nzchar(d)) "npi_redistribution" else d
}
source(file.path(here(), "plot_style.R"))

roll7 <- function(x) {
  n <- length(x)
  if (n < 3) return(x)
  k <- min(7, if (n %% 2 == 0) n - 1 else n)
  h <- (k - 1) %/% 2
  vapply(seq_len(n), function(i)
    mean(x[max(1, i - h):min(n, i + h)], na.rm = TRUE), numeric(1))
}

f <- file.path(in_dir, "calibration_fit.csv")
if (!file.exists(f)) stop(f, " missing")
d <- read.csv(f, stringsAsFactors = FALSE)
if (!"phase" %in% names(d)) d$phase <- "fitted"

d <- d %>%
  mutate(date = as.Date(date),
         w = factor(wave_label(wave, "short"), levels = wave_label(sort(unique(wave)), "short")),
         stream = factor(stream, levels = c("Detected cases", "Hospital admissions", "Deaths"))) %>%
  arrange(w, stream, date) %>%
  group_by(w, stream) %>%
  mutate(
    model_s    = if ("model_s" %in% names(.)) model_s else roll7(model),
    observed_s = roll7(observed),
    lo_s = if ("model_lo" %in% names(.)) model_lo else NA_real_,
    hi_s = if ("model_hi" %in% names(.)) model_hi else NA_real_) %>%
  ungroup()

has_band <- any(is.finite(d$lo_s))
gof_f <- file.path(in_dir, "calibration_runs.txt")
n_runs_txt <- if (file.exists(gof_f)) trimws(readLines(gof_f, warn = FALSE)[1]) else "repeated"

burn_rows <- d %>% filter(phase == "burn-in")
burn <- if (nrow(burn_rows) > 0) {
  burn_rows %>%
    group_by(w, stream) %>%
    summarise(xmin = min(date) - 0.5, xmax = max(date) + 0.5, .groups = "drop")
} else {
  burn_rows[0, c("w", "stream")] %>% mutate(xmin = as.Date(character()), xmax = as.Date(character()))
}

p <- ggplot(d, aes(date))
if (nrow(burn) > 0) {
  p <- p + geom_rect(data = burn, inherit.aes = FALSE,
                     aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
                     fill = BAND, colour = NA)
}
if (has_band) {
  p <- p + geom_ribbon(aes(ymin = lo_s, ymax = hi_s),
                       fill = CAT[1], alpha = 0.22, colour = NA, na.rm = TRUE)
}
p <- p +
  geom_point(aes(y = observed), colour = OBS, alpha = 0.22, size = 0.5, stroke = 0,
             na.rm = TRUE) +
  geom_line(aes(y = observed_s), colour = OBS, linewidth = 0.45, na.rm = TRUE) +
  geom_line(aes(y = model_s), colour = CAT[1], linewidth = 0.55, na.rm = TRUE) +
  facet_wrap(if (nlevels(d$w) > 1) vars(w, stream) else vars(stream),
             scales = "free", ncol = max(1, nlevels(d$w)), dir = "v") +
  scale_x_date(date_labels = "%d %b") +
  scale_y_continuous(labels = function(x) format(x, big.mark = ",", trim = TRUE)) +
  labs(x = NULL, y = "Count per day") +
  theme_paper() +
  theme(strip.text = element_text(size = ANNOT_PT, colour = INK),
        panel.spacing.x = unit(12, "pt"), panel.spacing.y = unit(8, "pt"),
        plot.margin = margin(4, 8, 3, 10))

cap2 <- c(
  "Both series are shown as a 7-day centred rolling mean; the observed daily counts are also",
  "plotted as points. Modelled cases are new infections scaled by the case-detection factor.",
  "Every panel carries its own vertical scale.")
if (has_band) {
  cap2 <- c(cap2, paste0(
    "The band is the 2.5-97.5th percentile of the Poisson predictive distribution over ",
    n_runs_txt, " runs of the calibrated model at fixed parameters: it combines the"),
    "simulator's own stochastic variation with observation noise. It is not a confidence",
    "interval on any parameter, and parameter uncertainty is not represented in it.",
    "The band narrows to nothing over the shaded days, where the series is read from the",
    "register rather than simulated.")
}
if (nrow(burn) > 0) {
  cap2 <- c(cap2,
            "Shaded days at the start of each window are excluded from the fit and from the",
            "calibration goodness-of-fit table: over them the series is routed from infections",
            "that occurred before the window opened, so it is set by the seeded initial",
            "condition rather than by transmission inside the window.")
}
cap2 <- c(cap2,
          "The calibration goodness-of-fit table reports the same days, computed on the",
          "UNSMOOTHED daily counts; smoothing raises the apparent agreement here.")

add_caption("figS_calibration", paste(
  "Supplementary Figure 1. The calibrated model against the series it was fitted to.",
  "Points and grey line: observed daily counts from the Danish registers.",
  "Coloured line: the calibrated model."), cap2)
save_fig("figS_calibration", p,
         width  = (30 + 75 * max(1, nlevels(d$w))) * MM,
         height = 210 * MM, out_dir)
write_captions(out_dir)
