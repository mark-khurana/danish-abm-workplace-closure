#!/usr/bin/env Rscript
# Figure 1: model structure, study period and prevented transmission by setting.
#   Rscript figures/R/figure1_context.R <data dir> <output dir> <fig1_settings.csv>

.args <- commandArgs(trailingOnly = TRUE)
ctx_dir <- if (length(.args) >= 1) .args[1] else "npi_redistribution/figure_context"
out_dir <- if (length(.args) >= 2) .args[2] else ctx_dir
val_csv <- if (length(.args) >= 3) .args[3] else ""

here <- function() {
  d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  if (is.na(d) || d == "") "." else d
}
source(file.path(here(), "plot_style.R"))
suppressPackageStartupMessages({ library(readr); library(cowplot); library(tidyr) })
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

LAYERS <- c(household = "Household", family = "Family",
            classmate = "School", colleague = "Workplace",
            neighbourhood = "Neighbourhood", mean_field = "Community")

W <- 1.05; PH <- 0.44; SHEAR <- 0.34
DX <- 0;   DY <- 0.58

base_pos <- tibble::tribble(
  ~node, ~bx,  ~by,
  1L,    0.22, 0.10,   2L,  0.34, 0.24,   3L,  0.26, 0.33,
  4L,    0.44, 0.12,   5L,  0.56, 0.26,   6L,  0.42, 0.35,
  7L,    0.72, 0.10,   8L,  0.84, 0.25,   9L,  0.70, 0.34,
  10L,   0.98, 0.13,  11L,  1.10, 0.27,  12L,  0.96, 0.36)

FOCUS <- 5L
CLUSTERS <- list(c(1, 2, 3), c(4, 5, 6), c(7, 8, 9), c(10, 11, 12))

tri <- function(v) list(c(v[1], v[2]), c(v[2], v[3]), c(v[1], v[3]))

edge_spec <- list(
  household     = do.call(c, lapply(CLUSTERS, tri)),
  family        = list(c(3, 6), c(6, 9), c(9, 12), c(1, 4), c(4, 7), c(7, 10),
                       c(2, 5), c(5, 8), c(8, 11)),
  classmate     = c(tri(c(1, 2, 4)), list(c(2, 4)),
                    tri(c(3, 6, 5)), list(c(5, 6)),
                    tri(c(7, 8, 10)), list(c(9, 12), c(11, 12), c(9, 11))),
  colleague     = list(c(1, 2), c(1, 3), c(1, 4), c(1, 5), c(1, 6), c(2, 3), c(2, 4), c(2, 5),
                       c(2, 6), c(3, 4), c(3, 5), c(3, 6), c(4, 5), c(4, 6), c(5, 6), c(7, 8),
                       c(7, 9), c(7, 10), c(7, 11), c(7, 12), c(8, 9), c(8, 10), c(8, 11),
                       c(8, 12), c(9, 10), c(9, 11), c(9, 12), c(10, 11), c(10, 12), c(11, 12)),
  neighbourhood = list(c(1, 4), c(4, 6), c(6, 9), c(7, 10), c(8, 11), c(5, 8),
                       c(2, 4), c(9, 11), c(10, 11), c(3, 5), c(1, 2), c(7, 12)),
  mean_field    = list()
)

plane_of <- function(i) list(dx = (i - 1) * DX, dy = (i - 1) * DY)

lv <- names(LAYERS)
planes <- bind_rows(lapply(seq_along(lv), function(i) {
  o <- plane_of(i)
  tibble::tibble(layer = lv[i], ord = i,
                 x = c(0, W, W + SHEAR, SHEAR) + o$dx,
                 y = c(0, 0, PH, PH) + o$dy)
}))

nodes6 <- bind_rows(lapply(seq_along(lv), function(i) {
  o <- plane_of(i)
  base_pos %>% mutate(layer = lv[i], ord = i, x = bx + o$dx, y = by + o$dy)
}))

edges6 <- bind_rows(lapply(seq_along(lv), function(i) {
  sp <- edge_spec[[lv[i]]]
  if (!length(sp)) return(NULL)
  p <- nodes6 %>% filter(layer == lv[i])
  do.call(rbind, lapply(sp, function(q) data.frame(
    layer = lv[i],
    x1 = p$x[p$node == q[1]], y1 = p$y[p$node == q[1]],
    x2 = p$x[p$node == q[2]], y2 = p$y[p$node == q[2]])))
}))

mf <- nodes6 %>% filter(layer == "mean_field")
mf_edges <- data.frame(
  x1 = mf$x[mf$node == FOCUS], y1 = mf$y[mf$node == FOCUS],
  x2 = mf$x[mf$node != FOCUS], y2 = mf$y[mf$node != FOCUS])

guides6 <- nodes6 %>% filter(node == FOCUS) %>% arrange(ord)

labs6 <- tibble::tibble(
  layer = lv, ord = seq_along(lv),
  title = unname(LAYERS[lv]),
  x = W + SHEAR + 0.06,
  y = (seq_along(lv) - 1) * DY + PH / 2)

HALO_GROUPS <- list(
  household = CLUSTERS,
  classmate = list(c(1, 2, 4), c(3, 5, 6), c(7, 8, 10), c(9, 11, 12)),
  colleague = list(c(1, 2, 3, 4, 5, 6), c(7, 8, 9, 10, 11, 12))
)

halo_ellipse <- function(px, py, pad = 0.042, n = 44) {
  cx <- mean(px); cy <- mean(py)
  rx <- max(abs(px - cx)) + pad
  ry <- max(abs(py - cy)) + pad * 0.85
  th <- seq(0, 2 * pi, length.out = n)
  x <- cx + rx * cos(th)
  y <- cy + ry * sin(th)
  data.frame(x = x + (y - cy) * (SHEAR / PH), y = y)
}

halos <- bind_rows(lapply(names(HALO_GROUPS), function(L) {
  i <- match(L, lv)
  p <- nodes6 %>% filter(layer == L)
  bind_rows(lapply(seq_along(HALO_GROUPS[[L]]), function(g) {
    v <- HALO_GROUPS[[L]][[g]]
    halo_ellipse(p$x[match(v, p$node)], p$y[match(v, p$node)]) %>%
      mutate(layer = L, grp = paste0(L, "_", g))
  }))
}))

panel_layers <- function() {
  ggplot() +
    geom_polygon(data = planes, aes(x, y, group = layer, fill = layer),
                 colour = RULE, linewidth = 0.25) +
    geom_polygon(data = halos, aes(x, y, group = grp), fill = "#ffffff",
                 alpha = 0.55, colour = NA) +
    geom_segment(data = mf_edges, aes(x = x1, y = y1, xend = x2, yend = y2),
                 colour = "#b4b4b4", linewidth = 0.22) +
    geom_segment(data = edges6, aes(x = x1, y = y1, xend = x2, yend = y2),
                 colour = "#4d4d4d", linewidth = 0.38) +
    geom_path(data = guides6, aes(x, y), colour = MUTE, linewidth = 0.35,
              linetype = "22") +
    geom_point(data = nodes6 %>% filter(node != FOCUS), aes(x, y),
               colour = INK, size = 1.8) +
    geom_point(data = nodes6 %>% filter(node == FOCUS), aes(x, y),
               colour = CAT[1], size = 2.5) +
    geom_text(data = labs6, aes(x, y, label = title), hjust = 0,
              size = ANNOT_MM, colour = INK) +
    scale_fill_manual(values = setNames(
      c(mako_at(seq(0.62, 0.86, length.out = length(lv) - 1)), "white"), lv),
      guide = "none") +
    coord_equal(xlim = c(-0.02, W + SHEAR + 0.80), clip = "off") +
    theme_void() +
    theme(plot.margin = margin(4, 2, 4, 4))
}

obs <- read_csv(file.path(ctx_dir, "observed_series.csv"), show_col_types = FALSE)
sched <- read_csv(file.path(ctx_dir, "npi_schedule.csv"), show_col_types = FALSE)

REPORTED <- if (nzchar(Sys.getenv("ABM_FIG_ALL_WAVES"))) {
  wave_meta$wave
} else {
  c("wave1")
}
EXCLUDED <- setdiff(wave_meta$wave, REPORTED)

obs <- obs %>% filter(!is.na(wave), wave != "", wave %in% REPORTED) %>%
  mutate(facet = factor(wave, levels = REPORTED))
if (!nrow(obs))
  stop("no observed rows for ", paste(REPORTED, collapse = ", "),
       ". observed_series.csv carries: ",
       paste(unique(read_csv(file.path(ctx_dir, "observed_series.csv"),
                             show_col_types = FALSE)$wave), collapse = ", "))

spans <- obs %>% group_by(wave) %>%
  summarise(start = min(date), end = max(date), .groups = "drop") %>%
  left_join(wave_meta, by = "wave") %>%
  mutate(lab = variant, facet = factor(wave, levels = REPORTED))

XBREAKS <- seq(as.Date("2020-12-01"), as.Date("2022-03-01"), by = "1 month")

date_scale <- function(labels = TRUE)
  scale_x_date(breaks = XBREAKS,
               labels = if (labels) en_month_year else NULL,
               expand = expansion(mult = 0.02))

wave_facet <- function()
  facet_grid(~ facet, scales = "free_x", space = "free_x",
             labeller = labeller(facet = setNames(spans$lab, spans$facet)))

panel_waves <- function() {
  ggplot() +
    geom_line(data = obs %>% filter(cases > 0), aes(date, cases),
              colour = INK, linewidth = 0.3) +
    wave_facet() +
    scale_y_continuous(labels = function(v) fmt_int(v),
                       expand = expansion(mult = c(0.02, 0.06))) +
    date_scale(labels = FALSE) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = "Reported cases per day") +
    theme_paper() +
    theme(axis.title.y = element_text(size = ANNOT_PT, angle = 90),
          axis.text.x = element_blank(),
          strip.text = strip_if_many(nrow(spans)),
          panel.grid.major.y = element_line(colour = RULE, linewidth = 0.3),
          panel.grid.minor.y = element_line(colour = "#eeeeee", linewidth = 0.2),
          plot.margin = margin(4, 8, 1, 4))
}

POLICY_ROWS <- c("school_closure", "workplace_closure", "gathering_ban", "mask_mandate")
gantt <- sched %>%
  mutate(row = ifelse(policy == "partial_workplace_closure",
                      "workplace_closure", policy)) %>%
  filter(row %in% POLICY_ROWS, days_active > 0, wave %in% REPORTED) %>%
  mutate(start = as.Date(start_date), end = as.Date(end_date),
         facet = factor(wave, levels = REPORTED),
         shade = ifelse(policy == "partial_workplace_closure", "#6f6f6f", INK),
         row = factor(relabel(row, POLICY_LABELS),
                      levels = rev(relabel(POLICY_ROWS, POLICY_LABELS))))
if (!nrow(gantt))
  stop("no NPI rows for ", paste(REPORTED, collapse = ", "),
       "; npi_schedule.csv has waves ", paste(unique(sched$wave), collapse = ", "))

panel_gantt <- function() {
  lv <- rev(relabel(POLICY_ROWS, POLICY_LABELS))
  ggplot() +
    geom_segment(data = gantt, aes(x = start, xend = end, y = row, yend = row,
                                   colour = shade),
                 linewidth = 3.4, lineend = "butt") +
    scale_colour_identity() +
    geom_text(data = gantt,
              aes(x = start + (end - start) / 2, y = row,
                  label = paste0(days_active, " d")),
              colour = "white", size = ANNOT_MM * 0.8) +
    scale_y_discrete(limits = lv, expand = expansion(add = 0.7)) +
    date_scale() +
    wave_facet() +
    labs(x = NULL, y = NULL) +
    theme_paper() +
    theme(axis.text.y = element_text(size = ANNOT_PT, hjust = 1),
          axis.text.x = element_text(size = ANNOT_PT, angle = 45, hjust = 1),
          axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          strip.text = element_blank(), strip.background = element_blank(),
          plot.margin = margin(1, 8, 3, 4))
}

panel_validation <- function(path) {
  v <- suppressWarnings(readr::read_csv(path, show_col_types = FALSE))
  need <- c("wave", "model_infection_rii", "model_infection_rii_lo",
            "model_infection_rii_hi", "observed_case_rii", "model_case_rii_uniform")
  miss <- setdiff(need, names(v))
  if (length(miss))
    stop(path, " is missing: ", paste(miss, collapse = ", "),
         " -- it must come from validation_table.py, not be hand-made")

  lv <- c("Observed (register)", "Model, cases", "Model, infections")
  d <- rbind(
    data.frame(wave = v$wave, series = lv[1], est = v$observed_case_rii,
               lo = NA_real_, hi = NA_real_),
    data.frame(wave = v$wave, series = lv[2], est = v$model_case_rii_uniform,
               lo = NA_real_, hi = NA_real_),
    data.frame(wave = v$wave, series = lv[3], est = v$model_infection_rii,
               lo = v$model_infection_rii_lo, hi = v$model_infection_rii_hi))
  d <- d[is.finite(d$est), ]
  if (!nrow(d)) stop(path, " has no finite gradient in any wave")
  d$series <- factor(d$series, levels = lv)
  d$wlab <- relabel(sub("^wave", "wave", d$wave), wave_labels)
  d$wlab <- factor(d$wlab, levels = rev(unname(wave_labels)))
  d$ypos <- as.integer(d$wlab) + seq(0.24, -0.24,
                                     length.out = length(lv))[as.integer(d$series)]

  brk <- c(0.6, 0.8, 1, 1.5, 2, 3)
  ggplot(d, aes(y = ypos, colour = series)) +
    geom_vline(xintercept = 1, colour = INK, linewidth = 0.4) +
    geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.5, na.rm = TRUE) +
    geom_point(aes(x = est), size = 1.4) +
    scale_colour_manual(values = setNames(PALETTE[c(6, 2, 1)], lv), name = NULL) +
    scale_x_continuous(trans = "log10", breaks = brk, labels = fmt_num(brk, 2),
                       limits = range(brk)) +
    scale_y_continuous(breaks = seq_len(nlevels(d$wlab)), labels = levels(d$wlab),
                       limits = c(0.45, nlevels(d$wlab) + 0.55), expand = c(0, 0)) +
    labs(x = "Ratio of infection rate, most to least deprived (log scale)",
         y = NULL) +
    theme_paper() +
    theme(legend.position = "bottom",
          legend.text = element_text(size = ANNOT_PT, colour = INK),
          legend.key.height = unit(3, "mm"),
          axis.title.x = element_text(size = ANNOT_PT, colour = INK),
          axis.text = element_text(size = ANNOT_PT, colour = INK))
}

FIG1_CAP <- c(
  "(a) The five contact layers on which the model transmits, with the community channel, drawn for twelve people.",
  "Layers are six settings in which the same 5\u00b78 million individuals meet, not six separate networks; the same",
  "twelve people appear on every layer in the same position, and one individual is traced through the stack in",
  "blue. Pale shading marks households, one school, and one workplace. Positions are illustrative; the",
  "layers and their structure are not.",
  paste0("(b) Reported cases per day and the interventions in force over the ",
         if (length(REPORTED) == 1) "period" else "periods",
         " reported. Bar length is the"),
  "number of days a policy was in force and does not indicate how strongly it applied. The paler bar marks a",
  "closure required in some sectors only. Schools were closed to all grades from Jan 4 to Feb 7, 2021, and open",
  "to some grades for the remainder of wave 1.",
  if (length(EXCLUDED))
    paste0("The x axis is truncated to ",
           if (length(REPORTED) == 1) "that period" else "those periods", ": the ",
           paste(gsub("\u2013", "-", wave_meta$variant[match(EXCLUDED, wave_meta$wave)]),
                 " (",
                 gsub("\u2013", "-", wave_meta$period[match(EXCLUDED, wave_meta$wave)]),
                 ")", sep = "", collapse = " and the "),
           if (length(EXCLUDED) == 1) " wave was" else " waves were",
           " simulated and excluded on the validity criteria given in the Methods;",
           " neither is interpreted here.")
  else "",
  "Bars are labelled with the number of days the policy was in force.")

SETTINGS_CSV <- if (length(.args) >= 3) .args[3] else "paper_figures/fig1_settings.csv"

LAYER_LAB <- c(colleague = "At work", household = "At home",
               family = "With family",
               classmate = "At school", neighborhood = "In the neighbourhood",
               `community(mean-field)` = "In the wider community")

panel_settings <- function(f) {
  d <- read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  if (!"transmission_ratio" %in% names(d) || all(is.na(d$transmission_ratio))) {
    message("  [note] ", f, " has no transmission_ratio -- panel C omitted. Run ",
            "paper_intervals.py, then export_paper.py.")
    return(NULL)
  }
  need <- c("transmission_ratio_lo", "transmission_ratio_hi")
  if (!all(need %in% names(d)))
    stop(f, " has transmission_ratio but not its interval. A bare point is not an ",
         "estimate; export_paper.py must carry lo/hi.")
  d <- d[!is.na(d$transmission_ratio), ]
  bad <- d$transmission_ratio < d$transmission_ratio_lo |
         d$transmission_ratio > d$transmission_ratio_hi
  if (any(bad))
    stop("panel C: ", sum(bad), " estimate(s) fall outside their own interval.")
  d$lab <- relabel(d$layer, LAYER_LAB)
  d$closed <- d$layer == "colleague"
  ord <- d[order(!d$closed, d$transmission_ratio), ]
  d$lab <- factor(d$lab, levels = rev(unique(ord$lab)))
  wl <- setNames(paste0("Wave ", match(sort(unique(d$wave)), sort(unique(d$wave))),
                        " (", c("wild-type/Alpha", "Omicron")[
                          match(sort(unique(d$wave)), c("wave1", "wave3"))], ")"),
                 sort(unique(d$wave)))
  d$w <- factor(unname(wl[as.character(d$wave)]), levels = unname(wl))
  ggplot(d, aes(transmission_ratio, lab, colour = closed)) +
    geom_vline(xintercept = 1, colour = INK, linewidth = 0.4) +
    geom_linerange(aes(xmin = transmission_ratio_lo, xmax = transmission_ratio_hi),
                   linewidth = 0.5) +
    geom_point(size = 1.5) +
    facet_wrap(~ w, nrow = 1) +
    scale_colour_manual(values = c(`TRUE` = CAT[1], `FALSE` = CAT[3]), guide = "none") +
    scale_x_continuous(trans = "log10", breaks = c(0.5, 0.7, 0.85, 1),
                       labels = function(z) fmt_num(z, 2),
                       expand = expansion(mult = c(0.07, 0.07))) +
    labs(x = "Infections acquired in each setting, with the workplace closure relative to without   (1 = unchanged)",
         y = NULL) +
    theme_paper() +
    theme(strip.text = strip_if_many(nlevels(d$w)),
          plot.margin = margin(4, 8, 3, 14))
}

top <- plot_grid(
  panel_layers(),
  plot_grid(panel_waves(), panel_gantt(), ncol = 1, align = "v", axis = "tblr",
            rel_heights = c(1, 0.72)),
  nrow = 1, rel_widths = c(0.33, 0.67),
  labels = c(panel_tag("A"), panel_tag("B")), label_size = 9,
  label_fontface = "bold", hjust = 0, label_x = 0.01, vjust = 1.1)

pc <- NULL
if (nzchar(Sys.getenv("ABM_FIG_VALIDATION")) && nzchar(val_csv) &&
    file.exists(val_csv)) {
  pc <- panel_validation(val_csv)
} else if (file.exists(SETTINGS_CSV)) {
  pc <- panel_settings(SETTINGS_CSV)
} else {
  message("  [note] no ", SETTINGS_CSV, " -- panels A and B only. Run ",
          "paper_intervals.py then export_paper.py to get panel C.")
}

if (!is.null(pc)) {
  fig <- plot_grid(top, pc, ncol = 1, rel_heights = c(1, 0.62),
                   labels = c("", panel_tag("C")), label_size = 9,
                   label_fontface = "bold", hjust = 0, label_x = 0.01, vjust = 1.4)
  add_caption("fig1", paste("Figure 1. Contact structure of the model (a), reported cases and interventions in",
                            "force over the modelled period(s) (b), and infections acquired in each setting",
                            "with and without the workplace closure (c)."),
              c(FIG1_CAP,
                "(c) Infections acquired in each setting with the workplace closure in force, divided by infections",
                "acquired in that setting without it; 1 indicates no change and values below 1 fewer infections. Each",
                "setting's infections are its share of the model's transmission pressure multiplied by that",
                "scenario's total infections; absolute per-setting counts are not recorded. Ratios are formed within",
                "each parameter draw. The workplace setting is shown in dark teal.",
                "School closure follows the same schedule in both scenarios, so school contact is identical in the",
                "ratio; differences at school arise from the proportion of contacts that were infectious. Infections",
                "acquired at school in wave 1 arise from the days on which schools were open to some grades.",
                "\"With family\" is contact with registered relatives.",
                "(c) covers the whole modelled population: per-setting transmission is recorded for the simulation as",
                "a whole and is not stratified by age, so unlike Figure 2 it cannot be restricted to working age.",
                INTERVAL_PHRASE))
  save_fig("figure1_what_we_built", fig, COL2, 138 * MM, out_dir)
} else {
  add_caption("fig1", paste("Figure 1. Contact structure of the model (a) and reported cases and interventions",
                            "in force over the modelled period(s) (b)."),
              FIG1_CAP)
  save_fig("figure1_what_we_built", top, COL2, 80 * MM, out_dir)
}
write_captions(out_dir)
message("  wrote ", file.path(out_dir, "figure1_what_we_built.pdf"))
