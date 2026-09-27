#!/usr/bin/env Rscript
# Figures 2 and 3: protection by workplace size, and income gradients.
#   Rscript figures/R/paper_figures.R <data dir> <output dir>

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(cowplot)
})

args    <- commandArgs(trailingOnly = TRUE)
in_dir  <- if (length(args) >= 1) args[1] else "paper_figures"
out_dir <- if (length(args) >= 2) args[2] else "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

here <- function() {
  d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  if (is.na(d) || !nzchar(d)) "npi_redistribution" else d
}
source(file.path(here(), "plot_style.R"))

need <- function(f) {
  p <- file.path(in_dir, f)
  if (!file.exists(p))
    stop(sprintf("%s is missing. Run export_paper.py (and paper_intervals.py before it) first.", p))
  read.csv(p, stringsAsFactors = FALSE, check.names = FALSE)
}

assert_intervals <- function(d, cols, what) {
  miss <- cols[!cols %in% names(d)]
  if (length(miss))
    stop(sprintf("%s: missing interval column(s) %s", what, paste(miss, collapse = ", ")))
  bad <- rowSums(is.na(d[cols])) > 0
  if (any(bad))
    stop(sprintf("%s: %d of %d rows have no interval. Run paper_intervals.py, or mark the quantity unidentified in export_paper.py -- do not plot a bare point.",
                 what, sum(bad), nrow(d)))
  invisible(TRUE)
}

assert_point_in_interval <- function(d, m, lo, hi, what) {
  v <- suppressWarnings(as.numeric(d[[m]]))
  a <- suppressWarnings(as.numeric(d[[lo]]))
  b <- suppressWarnings(as.numeric(d[[hi]]))
  ok <- is.finite(v) & is.finite(a) & is.finite(b)
  bad <- ok & (v < a | v > b)
  if (any(bad))
    stop(sprintf("%s: %d estimate(s) fall OUTSIDE their own interval, e.g. %.4f [%.4f, %.4f]. The point and the interval were computed from different draws or different quantities.",
                 what, sum(bad), v[bad][1], a[bad][1], b[bad][1]))
  span <- b - a
  edge <- ok & span > 0 & ((v - a) / span > 0.98 | (v - a) / span < 0.02)
  if (any(edge))
    message(sprintf("  [check] %s: %d estimate(s) sit within 2%% of an interval end, e.g. %.4f [%.4f, %.4f]. Legitimate for a skewed ratio; otherwise the median and the interval disagree.",
                    what, sum(edge), v[edge][1], a[edge][1], b[edge][1]))
  invisible(TRUE)
}

pct <- function(v) sprintf("%.0f%%", v * 100)

SHORT_VARIANT <- c(wave1 = "wild-type/Alpha", wave2 = "Alpha/Delta", wave3 = "Omicron")

wave_map <- function(waves) {
  u <- sort(unique(as.character(waves)))
  miss <- u[!u %in% names(SHORT_VARIANT)]
  if (length(miss))
    stop("wave_map(): no short name for ", paste(miss, collapse = ", "),
         ". Add it to SHORT_VARIANT.")
  setNames(paste0("Wave ", seq_along(u), " (", unname(SHORT_VARIANT[u]), ")"), u)
}
wave_full <- function(wmap)
  paste0(unname(wmap), ", ", wave_label(names(wmap), "period"))

fig2 <- function() {
  f1 <- need("fig2_protection.csv")
  hd <- need("headline.csv")
  lk <- need("fig2_leakage.csv")
  assert_intervals(f1, c("prot_frac_lo", "prot_frac_hi"), "fig2A protection")
  assert_point_in_interval(f1, "prot_frac", "prot_frac_lo", "prot_frac_hi", "fig2A")
  assert_point_in_interval(f1, "attack_A", "attack_A_lo", "attack_A_hi", "fig2B")
  assert_intervals(lk, c("averted_per_pop_share_lo", "averted_per_pop_share_hi"),
                   "fig2C leakage")
  assert_point_in_interval(lk, "averted_per_pop_share", "averted_per_pop_share_lo",
                           "averted_per_pop_share_hi", "fig2C")
  if (all(is.na(f1$attack_A)))
    stop("fig2: attack_A is entirely NA -- panel B IS the argument, so a missing baseline burden is a stop, not a degraded figure.")

  f1$band <- band_factor(f1$band)
  wmap <- wave_map(f1$wave)
  f1$w <- factor(unname(wmap[as.character(f1$wave)]), levels = unname(wmap))
  lk$band <- band_factor(lk$band)
  lk$w <- factor(unname(wmap[as.character(lk$wave)]), levels = unname(wmap))

  annot <- function(mcol, locol, hicol, from_metric) {
    rows <- lapply(unique(f1$wave), function(w) {
      h <- hd[hd$wave == w, ]
      if (!nrow(h) || is.null(h[[mcol]]) || is.na(h[[mcol]][1])) return(NULL)
      if (from_metric) {
        e <- trimws(strsplit(sub("^exposure_protection_ratio:", "", h$metric[1]),
                             "/", fixed = TRUE)[[1]])
        a <- tickify(e[1]); b <- tickify(e[2])
      } else {
        a <- tickify(levels(f1$band)[nlevels(f1$band)])
        b <- tickify(levels(f1$band)[1])
      }
      data.frame(w = unname(wmap[as.character(w)]),
                 lab = sprintf("%s vs %s:  %s  (%s to %s)", a, b,
                               fmt_num(h[[mcol]][1]), fmt_num(h[[locol]][1]),
                               fmt_num(h[[hicol]][1])))
    })
    rows <- do.call(rbind, rows)
    if (!is.null(rows)) rows$w <- factor(rows$w, levels = levels(f1$w))
    rows
  }
  annA <- annot("median", "lo", "hi", TRUE)
  annB <- if ("burden_median" %in% names(hd))
            annot("burden_median", "burden_lo", "burden_hi", FALSE) else NULL

  vpanel <- function(yv, ylo, yhi, ytitle, col, ann, showx, zero) {
    p <- ggplot(f1, aes(band, .data[[yv]]))
    if (zero) p <- p + geom_hline(yintercept = 0, colour = RULE, linewidth = 0.3)
    p <- p +
      geom_linerange(aes(ymin = .data[[ylo]], ymax = .data[[yhi]]),
                     linewidth = 0.5, colour = col, na.rm = TRUE) +
      geom_point(size = 1.7, colour = col) +
      facet_wrap(~ w, nrow = 1, scales = "free_y") +
      scale_x_discrete(labels = tickify) +
      scale_y_continuous(labels = pct,
                         expand = expansion(mult = c(if (zero) 0.05 else 0.12, 0.26))) +
      labs(x = if (showx) BAND_AXIS else NULL,
           y = ytitle) +
      theme_paper() +
      theme(strip.text = strip_if_many(nlevels(f1$w)),
            axis.title.y = element_text(size = ANNOT_PT, angle = 90),
            axis.text.x = if (showx) element_text(size = ANNOT_PT, colour = INK)
                          else element_blank(),
            plot.margin = margin(4, 8, 3, 16))
    if (!is.null(ann))
      p <- p + geom_text(data = ann, aes(x = 0.7, y = Inf, label = lab),
                         inherit.aes = FALSE, hjust = 0, vjust = 1.7,
                         size = ANNOT_MM, colour = MUTE)
    p
  }

  A <- vpanel("prot_frac", "prot_frac_lo", "prot_frac_hi",
              "Share of the group's own\nexpected infections prevented",
              CAT[1], annA, FALSE, TRUE)
  B <- vpanel("baseline_attack", "baseline_attack_lo", "baseline_attack_hi",
              "Share of the group infected\nif workplaces had stayed open",
              CAT[3], annB, FALSE, TRUE)

  lim <- max(abs(log(c(lk$averted_per_pop_share_lo, lk$averted_per_pop_share_hi)))) * 1.15
  C <- ggplot(lk, aes(band, averted_per_pop_share)) +
    geom_hline(yintercept = 1, colour = INK, linewidth = 0.4) +
    geom_linerange(aes(ymin = averted_per_pop_share_lo, ymax = averted_per_pop_share_hi),
                   linewidth = 0.7, colour = CAT[1]) +
    geom_point(size = 1.5, colour = CAT[1]) +
    facet_wrap(~ w, nrow = 1) +
    scale_x_discrete(labels = tickify) +
    scale_y_continuous(trans = "log10", limits = exp(c(-lim, lim)),
                       breaks = c(0.25, 0.5, 1, 2),
                       labels = function(z) fmt_num(z, 2)) +
    labs(x = BAND_AXIS,
         y = "Share of all prevented infections\n\u00f7 share of all people") +
    theme_paper() +
    theme(strip.text = strip_if_many(nlevels(lk$w)),
          axis.title.y = element_text(size = ANNOT_PT, angle = 90),
          plot.margin = margin(4, 8, 3, 16))

  p <- plot_grid(A, B, C, ncol = 1, align = "v", axis = "lr",
                 labels = c(panel_tag("A"), panel_tag("B"), panel_tag("C")),
                 label_size = 9, label_fontface = "bold", label_x = 0.004)

  add_caption("fig2", paste("Figure 2. Share of expected infections prevented by the workplace closure (a),",
                            "infection risk with workplaces open (b), and share of all prevented infections",
                            "relative to population share (c), by number of colleagues."), c(
    "(a) The share of the infections each group would have had, had workplaces remained open, that the workplace",
    "closure prevented, by the number of colleagues the individual holds in the register contact network.",
    "(b) The same groups' risk of infection with workplaces open. The concentration per 100 000 people is the",
    "product of (a) and (b). The vertical scale in (b) does not start at zero, so that the gradient between groups",
    "is visible; the scales in (a) and (c) include their reference value.",
    "(c) Each group's share of all prevented infections divided by its share of the population; 1 indicates a",
    "proportionate share.",
    "(a) and (b) are annotated with the ratio between the two end groups, differenced within each parameter draw;",
    "these ratios are narrower than the overlap of the plotted intervals suggests.", INTERVAL_PHRASE,
    pop_note(rbind(f1["stratifier"], lk["stratifier"]), "throughout"),
    paste0("Waves: ", paste(wave_full(wmap), collapse = "; "), "."),
    if (length(wmap) > 1)
      c("Wave numbering refers to the waves reported here; the remaining modelled waves are",
        "excluded on the validity criteria given in the Methods and are not interpreted.")
    else
      paste("Denmark's other modelled waves are excluded on the validity criteria given in",
            "the Methods and are not interpreted here.")))

  save_fig("fig2_protection", p, COL2, 152 * MM, out_dir)
}

fig3 <- function() {
  d <- need("fig3_income.csv")
  assert_intervals(d, c("lo", "hi"), "fig3 income")
  assert_point_in_interval(d, "median", "lo", "hi", "fig3 income")

  pre <- "RII_attack_B" %in% d$metric
  PANELS <- c(
    RII_attack_B = "Infection risk without the closure   (left of 1: less infection among the poorest)",
    RII_attack_A = "Infection risk with the closure in force   (left of 1: less infection among the poorest)",
    RII_prot     = "Protection by the closure   (left of 1: less protection among the poorest)")
  PANELS <- PANELS[names(PANELS) != if (pre) "RII_attack_A" else "RII_attack_B"]
  if (!pre)
    message("  [note] fig3 panel A is arm A, the world WITH the closure. RII_attack_B is ",
            "absent, so the mediation claim cannot be made cleanly from this render.")

  moved <- sum(d$metric == "RII_disp")
  d <- d[d$metric %in% names(PANELS), ]
  if (!nrow(d)) stop("fig3: no infection or protection index present.")
  if (moved)
    message("  [note] fig3: the ", moved, " displacement rows are not plotted. They are ",
            "a null and belong in the appendix; the CSV still carries them.")

  FITS <- c("As measured"                      = "Unadjusted",
            "Age held constant"                = "Adjusted for age",
            "Workplace exposure held constant" = "Adjusted for workplace attachment",
            "Both held constant"               = "Adjusted for both")
  d$fitlab <- relabel(d$fit, FITS)
  ordered_fits <- unique(d$fit[order(d$fit_order)])
  lv <- unname(FITS[names(FITS) %in% ordered_fits])
  lv <- lv[order(match(names(FITS)[names(FITS) %in% ordered_fits], ordered_fits))]
  d$fitlab <- factor(d$fitlab, levels = lv)
  nf <- length(lv)
  d$y <- nf + 1L - as.integer(d$fitlab)
  wmap <- wave_map(d$wave)
  d$w <- factor(unname(wmap[as.character(d$wave)]), levels = unname(wmap))

  waves <- sort(unique(d$wave))
  off <- if (length(waves) == 1) setNames(0, waves)
         else setNames(seq(0.16, -0.16, length.out = length(waves)), waves)
  d$ypos <- d$y + off[d$wave]

  XB <- c(0.5, 0.7, 1, 1.5, 2)
  need_lim <- max(abs(log(c(d$lo, d$hi))))
  lim <- min(log(XB)[log(XB) >= need_lim * 0.999], na.rm = TRUE)
  if (!is.finite(lim)) lim <- need_lim * 1.04
  x_hi  <- exp(lim) * 1.42
  x_lab <- exp(lim) * 1.06

  panel <- function(metric, showx) {
    dd <- d[d$metric == metric, ]
    ggplot(dd, aes(median, ypos, colour = w)) +
      geom_hline(yintercept = seq(1.5, nf - 0.5, 1), colour = RULE, linewidth = 0.25) +
      geom_vline(xintercept = 1, colour = INK, linewidth = 0.4) +
      geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.5) +
      geom_point(size = 1.6) +
      geom_text(aes(x = x_lab,
                    label = paste0(fmt_num(median, 2), "  (", fmt_num(lo, 2),
                                   "-", fmt_num(hi, 2), ")")),
                hjust = 0, size = ANNOT_MM, colour = INK, show.legend = FALSE) +
      scale_y_continuous(breaks = seq_len(nf), labels = rev(lv),
                         limits = c(0.4, nf + 0.6), expand = c(0, 0)) +
      scale_x_continuous(trans = "log10", limits = c(exp(-lim), x_hi),
                         breaks = XB, labels = function(z) fmt_num(z, 1)) +
      scale_colour_manual(values = setNames(CAT[c(1, 3)][seq_along(levels(d$w))],
                                            levels(d$w)), name = NULL) +
      labs(y = NULL,
           x = if (showx)
                 "Relative index of inequality, poorest end vs richest end (log scale)"
               else NULL,
           title = PANELS[[metric]]) +
      theme_paper() +
      theme(plot.title = element_text(size = ANNOT_PT, colour = INK, hjust = 0,
                                      face = "plain"),
            plot.title.position = "panel",
            plot.margin = margin(4, 8, 3, 12),
            axis.text.y = element_text(size = ANNOT_PT),
            axis.text.x = if (showx) element_text(size = ANNOT_PT, colour = INK)
                          else element_blank(),
            legend.position = if (showx && length(waves) > 1) "bottom" else "none",
            legend.text = element_text(size = ANNOT_PT, colour = INK),
            legend.key.height = unit(8, "pt"))
  }

  mets <- names(PANELS)[names(PANELS) %in% d$metric]
  ps <- lapply(seq_along(mets), function(k) panel(mets[k], showx = k == length(mets)))
  p <- plot_grid(plotlist = ps, ncol = 1, align = "v", axis = "lr",
                 rel_heights = c(rep(1, length(ps) - 1), 1.22),
                 labels = letters[seq_along(ps)], label_size = 9,
                 label_fontface = "bold", label_x = 0.004)

  add_caption("fig3",
              paste("Figure 3. Relative index of inequality by household income for infection without the",
                    "workplace closure (a) and for protection by the closure (b), under successive adjustments."), c(
    "Both panels compare the poorest and richest ends of the household-income distribution among people alike in",
    "the way each row states. The measure is the relative index of inequality, fitted across all four income",
    "quartiles rather than a raw Q1-to-Q4 ratio, so 1 means no difference.",
    "(a) Infections each group would have had in the scenario in which workplaces remained open. The difference",
    "between income groups is attenuated when people with the same number of colleagues are compared.",
    "(b) The share of those infections prevented by the closure, comparing that scenario with a closure in which",
    "displaced contact was switched off. The difference between income groups persists under the same adjustment.",
    "Figure 1c compares the observed epidemic with the workplaces-open scenario. Household infections were lower",
    "with the closure than without it; displaced contact increased them.",
    INTERVAL_PHRASE, pop_note(d, "in every row of both panels"),
    paste0("Waves: ", paste(wave_full(wmap), collapse = "; "), ".")))

  save_fig("fig3_income", p, COL2, 132 * MM, out_dir)
}

fig2(); fig3()
write_captions(out_dir)
message("\n  Rendered into ", out_dir, ".",
        "\n  NO text is printed on the artwork -- the captions are in ", out_dir,
        "/captions.md.")
