"""Seasonal transmission forcing for the Danish ABM."""

from dataclasses import dataclass, field
from typing import Optional, List

import torch
import numpy as np
import math


@dataclass
class SeasonalForcing:
    """Seasonal transmission multiplier."""
    kind: str = "sinusoidal"
    peak_day: int = 15
    amplitude: float = 0.3
    baseline: float = 1.0
    values: Optional[np.ndarray] = None
    school_amplitude: float = 0.0
    school_holidays: Optional[List[tuple]] = None

    def __call__(self, day: int, sim_start_doy: int = 0) -> float:
        """Return the seasonal multiplier for a given simulation day."""
        doy = (sim_start_doy + day) % 365

        if self.kind == "none":
            return 1.0

        if self.kind == "sinusoidal":
            phase = 2.0 * math.pi * (doy - self.peak_day) / 365.0
            mult = self.baseline + self.amplitude * math.cos(phase)

        elif self.kind == "school_calendar":
            phase = 2.0 * math.pi * (doy - self.peak_day) / 365.0
            mult = self.baseline + self.amplitude * math.cos(phase)
            if self.school_holidays:
                in_holiday = any(s <= doy < e for s, e in self.school_holidays)
                if in_holiday:
                    mult -= self.school_amplitude
                else:
                    mult += self.school_amplitude * 0.2

        elif self.kind == "data":
            if self.values is None:
                return 1.0
            if len(self.values) == 12:
                month = doy // 30
                month = min(month, 11)
                return float(self.values[month])
            else:
                return float(self.values[doy % len(self.values)])

        else:
            return 1.0

        return max(0.0, mult)

    def build_tensor(self, n_days: int, sim_start_doy: int = 0) -> torch.Tensor:
        """Build a (n_days,) tensor of seasonal multipliers."""
        return torch.tensor([self(d, sim_start_doy) for d in range(n_days)],
                            dtype=torch.float32)

    def describe(self) -> str:
        if self.kind == "none":
            return "No seasonal forcing"
        if self.kind == "sinusoidal":
            return (f"Sinusoidal: peak DOY={self.peak_day}, "
                    f"amplitude={self.amplitude:.0%}, range=[{self.baseline-self.amplitude:.2f}, {self.baseline+self.amplitude:.2f}]")
        if self.kind == "school_calendar":
            return (f"School calendar: peak DOY={self.peak_day}, "
                    f"amplitude={self.amplitude:.0%}, school_amp={self.school_amplitude:.0%}")
        if self.kind == "data":
            n = len(self.values) if self.values is not None else 0
            return f"Data-driven: {n} values"
        return f"{self.kind}"


DENMARK_SCHOOL_HOLIDAYS = [
    (0, 3),
    (42, 49),
    (88, 95),
    (166, 222),
    (285, 292),
    (355, 365),
]

SEASONAL_PRESETS = {
    "none": SeasonalForcing(kind="none"),
    "covid_mild": SeasonalForcing(kind="sinusoidal", peak_day=15, amplitude=0.15),
    "influenza": SeasonalForcing(
        kind="school_calendar", peak_day=30, amplitude=0.35,
        school_amplitude=0.1, school_holidays=DENMARK_SCHOOL_HOLIDAYS,
    ),
    "rsv": SeasonalForcing(kind="sinusoidal", peak_day=350, amplitude=0.40),
}


def get_seasonal(name: str) -> SeasonalForcing:
    if name not in SEASONAL_PRESETS:
        raise KeyError(f"Unknown seasonal preset '{name}'. Available: {list(SEASONAL_PRESETS.keys())}")
    return SEASONAL_PRESETS[name]
