# Experiment Config Reference

This document describes each field in `configs/experiment.json`, what it controls, and common/expected values.

## Top-level sections

- `experiment`: Controls how many runs and how scenarios are combined.
  - `mode`: `"single"` or `"multi"`.
    - `single`: run exactly one simulation using the other top-level defaults.
    - `multi`: iterate over `conditions` × `seeds`/`num_runs`.
  - `name`: Free-form label for your experiment (no functional effect).
  - `seeds`: List of integers; each seed produces one run per condition. If omitted, `num_runs` is used.
  - `num_runs`: Integer fallback when `seeds` is not provided; creates that many runs with `seed=None`.
  - `conditions`: List of condition objects; each can override `environment` and/or `lkas`.
    - Each condition object may include:
      - `environment`: Same shape as the top-level `environment` section; values here override defaults.
      - `lkas`: Same shape as the top-level `lkas` section; values here override defaults.

- `simulation`: Global simulation defaults.
  - `map`: CARLA town identifier, e.g. `"Town04"`, `"Town10HD"`, `"Town03"`.
  - `distance_meters`: Float/int distance before stopping, e.g. `1000`.
  - `seed`: Optional fixed random seed for the simulation (not commonly used; prefer `experiment.seeds`).

- `environment`: Scenario defaults.
  - `target_speed`: Desired speed in km/h (e.g. `80`, `90`).
  - `weather`: Preset key from `configs/weather_presets.json` (e.g. `"ClearNoon"`, `"MidFogNoon"`, `"HardRainNoon"`, `"ClearNight"`, `"SmallRainNoon"`, `"HeavyFogNoon"`).
  - `street_light`: `null`, `"on"`, or `"off"`; used when weather includes `"Night"`.
  - `vehicle_light`: `null`, `"LowBeam"`, or `"HighBeam"`; used when weather includes `"Night"`.

- `lkas`: Lane keeping stack settings.
  - `detector`: `"legacy"` or `"lanenet"`.

- `display`: Visualization settings.
  - `show_hud`: `true`/`false`; when `false`, HUD is not constructed or updated.

- `logging`: Output settings.
  - `enabled`: `true`/`false`; when `false`, no CSV output is written.

## How multi-run execution works

For each seed in `experiment.seeds` (or each placeholder run if only `num_runs` is provided), and for each condition in `experiment.conditions`, the simulator merges the condition overrides onto the top-level `environment`/`lkas` defaults and runs one simulation with that merged configuration.

## File naming for outputs

When logging is enabled, trajectory CSVs are named:

```
<detector>_<map>_<weather>_<target_speed>_<street_light>_<vehicle_light>[_seedN]_trajectory.csv
```

Where `_seedN` is included only when a seed is present.
