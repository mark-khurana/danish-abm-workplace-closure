#!/usr/bin/env Rscript
# Shared plotting style: colours, fonts, panel tags and number formatting.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

MM   <- 1 / 25.4
COL1 <- 90 * MM
COL2 <- 190 * MM
MAXH <- 240 * MM

INK  <- "#1a1a1a"
RULE <- "#d9d9d9"
MUTE <- "#6b6b6b"
BAND <- "#f0f0f0"

PALETTE <- c("#01665E", "#35978F", "#80CDC1", "#C7EAE5", "#F5F5F5",
             "#F6E8C3", "#DFC27D", "#BF812D", "#8C510A")

pal_at <- function(p) colorRampPalette(PALETTE, space = "Lab")(1001)[
  pmax(1, pmin(1001, round(p * 1000) + 1))]

CAT <- PALETTE[c(1, 9, 2, 8, 7)]
PALETTE <- CAT

OBS <- "#4D4D4D"

MAKO_STOPS <- c("#0d4a49", "#16605f", "#2a5674", "#3b738f", "#4f90a6", "#68abb8",
                "#85c4c9", "#a8dbd9", "#c4e7e3", "#d1eeea", "#e8f6f5")
mako_at <- function(p) colorRampPalette(MAKO_STOPS, space = "Lab")(1001)[
  pmax(1, pmin(1001, round(p * 1000) + 1))]

ANNOT_PT <- 7
ANNOT_MM <- 2.5

theme_paper <- function(base_size = 9) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position  = "none",
      panel.grid.minor = element_blank(),
      panel.grid.major = element_blank(),
      panel.border     = element_rect(colour = RULE, fill = NA, linewidth = 0.4),
      axis.line        = element_blank(),
      axis.ticks       = element_line(colour = RULE, linewidth = 0.3),
      axis.ticks.length = unit(1.2, "pt"),
      axis.text        = element_text(size = ANNOT_PT, colour = INK),
      axis.title       = element_text(size = ANNOT_PT, colour = INK),
      plot.title       = element_text(size = base_size, face = "bold", hjust = 0),
      plot.title.position = "plot",
      plot.margin      = margin(4, 8, 3, 4)
    )
}

panel_tag <- function(letter) tolower(letter)

fmt_num <- function(x, digits = 2)
  formatC(x, format = "f", digits = digits)

fmt_int <- function(x) formatC(round(x), format = "d", big.mark = " ")

strip_if_many <- function(n)
  if (n > 1) element_text(size = ANNOT_PT, colour = INK) else element_blank()

AXIS_LABELS <- c(
  benefit_per_100k = "Infections prevented per 100 000",
  cost_per_100k    = "Infections caused elsewhere per 100 000",
  prot_frac        = "Infections prevented (% of those expected)",
  disp_frac        = "Benefit cancelled out (%)",
  attack_A         = "Infections per person"
)

STRATUM_LABELS <- c(
  "0" = "No workplace", "1" = "Small workplace",
  "2" = "Medium workplace", "3" = "Large workplace",
  "0.0" = "No workplace", "1.0" = "Small workplace",
  "2.0" = "Medium workplace", "3.0" = "Large workplace",
  "0 (none)" = "No workplace", "1 (small)" = "Small workplace",
  "2 (medium)" = "Medium workplace", "3 (large)" = "Large workplace",
  "Q1" = "Q1 (lowest income)", "Q2" = "Q2", "Q3" = "Q3", "Q4" = "Q4 (highest)",
  "T1_outside_workforce"   = "Outside the workforce",
  "T2_unemployed"          = "Unemployed",
  "T3_employed"            = "Employed, non-managerial",
  "T4_manager_high_level"  = "Managers and senior professionals"
)

POLICY_LABELS <- c(
  school_closure            = "School closure",
  workplace_closure         = "Workplace closure",
  partial_workplace_closure = "Workplace closure (partial)",
  gathering_ban             = "Gathering ban",
  mask_mandate              = "Mask mandate",
  social_distancing         = "Stay-at-home order"
)

relabel <- function(x, lookup) {
  x <- as.character(x)
  ifelse(x %in% names(lookup), unname(lookup[x]), x)
}

wave_meta <- tibble::tribble(
  ~wave,   ~short,   ~period,                     ~variant,             ~vaccination,
  "wave1", "Wave 1", "16 Dec 2020 – 28 Feb 2021", "wild-type to Alpha", "<5% covered",
  "wave2", "Wave 2", "1 Mar 2021 – 10 Sep 2021",  "Alpha to Delta",     "mass rollout",
  "wave3", "Wave 3", "11 Nov 2021 – 1 Feb 2022",  "Delta to Omicron BA.1", "boosters"
)
wave_labels <- setNames(wave_meta$short, wave_meta$wave)

wave_label <- function(w, form = c("full", "short", "period")) {
  form <- match.arg(form)
  i <- match(as.character(w), wave_meta$wave)
  if (any(is.na(i)))
    stop("wave_label(): unknown wave ", paste(unique(w[is.na(i)]), collapse = ", "),
         ". Known: ", paste(wave_meta$wave, collapse = ", "))
  per <- gsub("\u2013", "-", wave_meta$period[i])
  switch(form,
         period = per,
         short  = gsub("\u2013", "-", wave_meta$variant[i]),
         full   = paste0(per, "  \u00b7  ", gsub("\u2013", "-", wave_meta$variant[i])))
}

BAND_TICKS <- c("0 (none)" = "No workplace", "1 (small)" = "Small workplace",
                "2 (medium)" = "Medium workplace", "3 (large)" = "Large workplace",
                "0" = "No workplace", "1" = "Small workplace",
                "2" = "Medium workplace", "3" = "Large workplace",
                "0.0" = "No workplace", "1.0" = "Small workplace",
                "2.0" = "Medium workplace", "3.0" = "Large workplace")

BAND_AXIS <- "Workplace size"

tickify <- function(x) ifelse(x %in% names(BAND_TICKS), unname(BAND_TICKS[x]), x)

band_factor <- function(x) {
  x <- as.character(x)
  key <- suppressWarnings(as.numeric(sub("^([-0-9.]+).*$", "\\1", x)))
  factor(x, levels = unique(x[order(key, x)]))
}

annot_column <- function(ypos, labels, header = NULL, ymax = NULL) {
  stopifnot(length(ypos) == length(labels))
  d <- data.frame(y = ypos, lab = as.character(labels))
  ggplot(d, aes(0, y)) +
    geom_text(aes(label = lab), size = ANNOT_MM, colour = INK, hjust = 0) +
    scale_x_continuous(limits = c(-0.02, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0.4, (ymax %||% max(ypos)) + 0.6), expand = c(0, 0)) +
    labs(x = NULL, y = NULL, title = if (is.null(header)) " " else header) +
    theme_void() +
    theme(plot.title = element_text(size = ANNOT_PT, colour = MUTE, hjust = 0,
                                   lineheight = 0.9))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

wrap_cap <- function(s, width = 108) paste(strwrap(s, width), collapse = "\n")

INTERVAL_PHRASE <- paste(
  "Bars span the 2.5th to 97.5th percentile of the parameter-draw ensemble.",
  "They describe how the model moves as its calibrated scales vary; they are not",
  "confidence or credible intervals and carry no coverage interpretation.")

pop_of <- function(d) {
  if ("population" %in% names(d)) {
    v <- unique(as.character(d$population[!is.na(d$population)]))
  } else if ("stratifier" %in% names(d)) {
    v <- unique(ifelse(grepl("_wa($|[^a-z])", as.character(d$stratifier)),
                       "working age", "all ages"))
  } else {
    return(NA_character_)
  }
  if (length(v) == 0) NA_character_ else v
}

pop_note <- function(d, where, require_wa = TRUE) {
  v <- pop_of(d)
  if (length(v) > 1)
    stop(sprintf(paste("caption: this panel mixes populations (%s). Reading down it the",
                       "population changes as well as the adjustment, so a narrowing",
                       "gradient cannot be attributed to the adjustment. Re-read with",
                       "read_workplace_band.py --restrict-age 2,3,4,5,6 --merge --refresh",
                       "before drafting."),
                 paste(v, collapse = " + ")))
  if (length(v) == 1 && is.na(v))
    stop(paste("caption: no `population` or `stratifier` column, so the population",
               "cannot be stated. A caption must not assert one the data does not carry."))
  if (require_wa && !identical(v, "working age"))
    stop(sprintf(paste("caption: this panel is on the %s population, but the paper's",
                       "estimand is working age (20-69). Either the read fell back to an",
                       "all-ages marginal or the wrong stratifier won -- fix the data, do",
                       "not relabel the figure."), v))
  if (identical(v, "working age"))
    sprintf("Working-age population, ages 20-69, %s.", where)
  else
    sprintf("Whole modelled population (all ages), %s.", where)
}

.CAPTIONS <- new.env(parent = emptyenv())

add_caption <- function(name, title, text) {
  assign(name, list(title = title, text = paste(text, collapse = " ")), envir = .CAPTIONS)
  invisible(TRUE)
}

write_captions <- function(out_dir, file = "captions.md") {
  nm <- sort(ls(.CAPTIONS))
  if (!length(nm)) return(invisible(FALSE))
  p <- file.path(out_dir, file)

  if (file.exists(p)) {
    old <- readLines(p, warn = FALSE)
    starts <- grep("^## ", old)
    for (i in seq_along(starts)) {
      key <- sub("^## ", "", old[starts[i]])
      if (key %in% nm) next
      end <- if (i < length(starts)) starts[i + 1] - 1 else length(old)
      body <- old[(starts[i] + 1):end]
      body <- body[nzchar(trimws(body))]
      if (!length(body)) next
      txt <- sub("^\\*\\*(.*?)\\*\\* ", "", body[1])
      ttl <- sub("^\\*\\*(.*?)\\*\\*.*$", "\\1", body[1])
      assign(key, list(title = ttl, text = txt), envir = .CAPTIONS)
    }
    nm <- sort(ls(.CAPTIONS))
  }

  con <- file(p, "w")
  writeLines(c("# Figure captions -- generated, copy from here",
               "",
               "Nothing here is printed on the artwork. Paste into the manuscript.",
               ""), con)
  for (n in nm) {
    cc <- get(n, envir = .CAPTIONS)
    writeLines(c(paste0("## ", n), "", paste0("**", cc$title, "** ", cc$text), ""), con)
  }
  close(con)
  message("  captions -> ", p)
  invisible(TRUE)
}

en_month_year <- function(x)
  ifelse(is.na(x), NA_character_,
         paste0(month.abb[as.integer(format(x, "%m"))], " ", format(x, "%Y")))

save_fig <- function(name, plot, width, height, out_dir) {
  if (height > MAXH)
    warning(sprintf("%s is %.0f mm deep; the limit is %.0f mm",
                    name, height / MM, MAXH / MM))
  dev <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot, device = dev,
         width = width, height = height, bg = "white")
  if (nzchar(Sys.getenv("ABM_FIG_PNG")))
    ggsave(file.path(out_dir, paste0(name, ".png")), plot,
           width = width, height = height, dpi = 300, bg = "white")
  message("  ", name, " saved")
}
