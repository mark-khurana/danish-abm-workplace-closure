# Figure 4: presence at work and protection by occupational group,
# under a uniform and a teleworkability-weighted reduction.
#   Rscript figures/R/figure4_jobtype.R <data dir> <output dir>
suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(cowplot) })
args    <- commandArgs(trailingOnly = TRUE)
in_dir  <- if (length(args) >= 1) args[1] else "paper_figures"
out_dir <- if (length(args) >= 2) args[2] else "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
here <- function() {
  d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  if (is.na(d) || !nzchar(d)) "npi_redistribution" else d
}
source(file.path(here(), "plot_style.R"))
need <- function(f) { p <- file.path(in_dir, f); if (!file.exists(p)) stop(p, " missing"); read.csv(p, stringsAsFactors = FALSE, check.names = FALSE) }
DRAFT <- nzchar(Sys.getenv("ABM_FIG4_DRAFT"))

RUNS <- c(uniform = "Every colleague link reduced equally",
          weighted = "Reduction weighted by the tier's teleworkability",
          literal = "Teleworkability taken literally, larger total fall")
COL  <- c(uniform = CAT[1], weighted = CAT[2], literal = PALETTE[7])
SHP  <- c(uniform = 16, weighted = 23, literal = 4)

d <- need("fig4_jobtype.csv") %>%
  mutate(run = factor(run, levels = names(RUNS)),
         w = factor(wave_label(wave, "short"), levels = wave_label(sort(unique(wave)), "short")))
runs_present <- levels(droplevels(d$run))

wrap_rows <- function(x, width = 34)
  vapply(x, function(s) paste(strwrap(s, width = width), collapse = "\n"), "", USE.NAMES = FALSE)

dumbbell <- function(dd, xlab, ref = NULL, pct_axis = FALSE, legend = FALSE) {
  dd <- dd %>% mutate(row = wrap_rows(row),
                      row = factor(row, levels = rev(unique(row[order(row_order)]))))
  p <- ggplot(dd, aes(estimate, row))
  if (!is.null(ref)) p <- p + geom_vline(xintercept = ref, colour = INK, linewidth = 0.4)
  p <- p +
    geom_hline(yintercept = seq(1.5, length(levels(dd$row)) - 0.5), colour = RULE, linewidth = 0.25) +
    geom_linerange(aes(xmin = lo, xmax = hi, colour = run), linewidth = 0.4, alpha = 0.45,
                   position = position_dodge(width = 0.6), na.rm = TRUE) +
    geom_point(aes(colour = run, shape = run, fill = run), size = 2.1, stroke = 0.9,
               position = position_dodge(width = 0.6)) +
    facet_wrap(~ w, nrow = 1, scales = "free_x") +
    scale_colour_manual(values = COL, labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
    scale_fill_manual(values = c(uniform = CAT[1], weighted = "white", literal = "white"),
                      labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
    scale_shape_manual(values = SHP, labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
    labs(x = xlab, y = NULL) +
    theme_paper() +
    theme(strip.text = strip_if_many(nlevels(dd$w)),
          axis.text.y = element_text(size = ANNOT_PT, colour = INK),
          plot.margin = margin(4, 8, 3, 10),
          legend.position = "none")
  if (legend)
    p <- p +
      guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1),
             fill = guide_legend(nrow = 1)) +
      theme(legend.position = "bottom", legend.justification = "left",
            legend.text = element_text(size = ANNOT_PT, margin = margin(l = 3, r = 10)),
            legend.key.size = unit(9, "pt"), legend.spacing.x = unit(4, "pt"),
            legend.margin = margin(2, 0, 0, 0))
  if (pct_axis) p <- p + scale_x_continuous(labels = function(x) paste0(round(100 * x), "%"))
  else p <- p + scale_x_continuous(trans = "log10", breaks = c(0.5, 0.7, 1, 1.4, 2),
                                   labels = function(z) fmt_num(z, 1))
  p
}

DROP <- c("Protection ratio, large workplace vs no workplace attachment",
          "   at fixed workplace attachment")
RELABEL <- c("Income gradient in protection, poorest vs richest" = "All adults aged 20-69")
p_tier <- dumbbell(filter(d, panel == "a"),
                   "Share of the group's expected infections that the closure prevented (%)", pct_axis = TRUE)
p_grad <- dumbbell(filter(d, panel == "b", !(row %in% DROP)) %>%
                     mutate(row = ifelse(row %in% names(RELABEL), RELABEL[row], row)),
                   "Ratio of protection, poorest to richest (log scale; below 1 = the poorest protected less)",
                   ref = 1, legend = TRUE)

pr <- need("fig4_presence.csv") %>%
  mutate(run = factor(run, levels = names(RUNS)),
         tier = factor(tier, levels = unique(tier[order(tier_order)])),
         w = factor(wave_label(wave, "short"), levels = levels(d$w)))
p_pres <- ggplot(pr, aes(presence, tier, colour = run, shape = run, fill = run)) +
  geom_vline(xintercept = 1, colour = RULE, linewidth = 0.3) +
  geom_hline(yintercept = seq(1.5, length(levels(pr$tier)) - 0.5), colour = RULE, linewidth = 0.25) +
  geom_point(size = 2.1, stroke = 0.9, position = position_dodge(width = 0.6)) +
  facet_wrap(~ w, nrow = 1) +
  scale_colour_manual(values = COL, labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
  scale_fill_manual(values = c(uniform = CAT[1], weighted = "white", literal = "white"),
                    labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
  scale_shape_manual(values = SHP, labels = RUNS[runs_present], name = NULL, breaks = runs_present) +
  scale_x_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0.5, 1.02),
                     breaks = c(0.5, 0.6, 0.7, 0.8, 0.9, 1), expand = c(0, 0)) +
  scale_y_discrete(limits = rev(levels(pr$tier))) +
  labs(x = "Presence at work per employee while the reduction was in force (%)", y = NULL) +
  theme_paper() +
  theme(legend.position = "none",
        strip.text = strip_if_many(nlevels(pr$w)),
        axis.text.y = element_text(size = ANNOT_PT, colour = INK),
        panel.spacing.x = unit(16, "pt"),
        plot.margin = margin(4, 8, 3, 10))

p <- plot_grid(p_pres, p_tier, p_grad, ncol = 1, rel_heights = c(0.8, 1.05, 1.15), align = "v", axis = "lr",
               labels = c(panel_tag("A"), panel_tag("B"), panel_tag("C")), label_size = 9,
               label_fontface = "bold", label_x = 0.004)
if (DRAFT) p <- ggdraw(p) + draw_label("DRAFT -- every number is MOCK", x = 0.99, y = 0.995,
                                       hjust = 1, vjust = 1, size = 7, colour = "red")
add_caption("fig4", "Figure 4. Job type changes how unequally the closure protected. The assumed presence at work by skill tier (a), the share of expected infections prevented in each tier (b), and the income gradient in that protection overall and within each tier (c), under a uniform reduction in workplace contact and under one weighted by occupational teleworkability.", c(
  "Filled teal points: the primary analysis, every colleague link reduced by the same factor. Hollow brown",
  "points: the same total reduction distributed by the teleworkability of each employee's skill tier",
  "(SOCIO13 131-134; JRC/Eurofound teleworkability by ISCO major group), so that professionals withdraw",
  "more and basic-skill employees less. The weighting acts on the colleague layer only: the displacement",
  "of contact into households is a policy-level mechanism, identical in both arms, so it does not track",
  "which tier withdrew.",
  "(a) Workplace contact kept while the reduction was in force. A colleague link survives only if both",
  "people are present, so the fall in contact is the square of presence: both series average to the same",
  paste0("exposure-weighted fall, to 62% of pre-closure workplace contact."),
  "(b) Share of expected infections prevented, employees aged 20-69, age-standardised.",
  "(c) Relative index of inequality (RII) by household income in protection, over all adults aged 20-69",
  "(people with no workplace attachment included) and within each skill tier; below 1 means the poorest",
  "were protected less than the richest. Conditioning on the tier blocks the path from income through job",
  "type, which is why the within-tier gradients sit closer to 1. Weighting by",
  "teleworkability moves every estimate further below 1, so the uniform assumption is the conservative one.",
  "The self-employed and ISCO 9 are excluded from the within-tier gradients, the first because employment",
  "status and household income are jointly determined, the second because it barely withdraws at all.",
  "The presence axis starts at 50%: no group falls below it in either arm.",
  "Ranges span the 2.5th to 97.5th percentile across 100 parameter draws and are not confidence intervals."))
save_fig("fig4_jobtype", p, width = 180 * MM, height = 215 * MM, out_dir)
write_captions(out_dir)
