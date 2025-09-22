# configs.py
from dataclasses import dataclass, field
from datetime import datetime
import yaml, re
from typing import Optional, Dict, Any

_RC_RE = re.compile(r"^(?P<year>\d{4})MF(?P<month>[1-9]|1[0-2])$")
DEFAULT_DATE_START = "2017-01-31 00:00:00"

def _parse_reporting_cycle(rc: str) -> tuple[int, int]:
    m = _RC_RE.match(rc.strip())
    if not m:
        raise ValueError(
            f"Invalid REPORTING_CYCLE: {rc!r}. "
            "Expected format like '2025MF6' (year + MF + month 1–12)."
        )
    return int(m.group("year")), int(m.group("month"))

@dataclass(frozen=True)
class RunParamsConfigs:
    REPORTING_CYCLE: str
    DATE_START: str = field(default=DEFAULT_DATE_START, init=False)
    DATE_CUT: str = field(init=False)
    FORECAST_HORIZON: int = field(init=False)

    def __post_init__(self):
        year, month = _parse_reporting_cycle(self.REPORTING_CYCLE)
        cut = datetime(year, 12, 31, 0, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
        horizon = 12 - month
        object.__setattr__(self, "DATE_CUT", cut)
        object.__setattr__(self, "FORECAST_HORIZON", horizon)

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "RunParamsConfigs":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            raw.update(overrides)
        rc = str(raw.get("reporting_cycle") or raw.get("REPORTING_CYCLE"))
        return cls(REPORTING_CYCLE=rc)

# --------- instantiate once ---------
class Configs:
    CurrentRunParams: RunParamsConfigs = RunParamsConfigs.from_yaml("run_params.yaml")