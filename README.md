# SimLKAS

Code for paper:  
**SimLKAS: a simulation-based framework for verification and validation of lane keeping assistance systems**.  
This repository accompanies the publication:  
[https://doi.org/10.1080/15472450.2025.2559224](https://doi.org/10.1080/15472450.2025.2559224)

---

## Description
SimLKAS provides a modular simulation framework to evaluate LKAS performance under various conditions, including adverse weather and complex road geometries. It integrates perception, control, and evaluation components for systematic testing.

---

## Project layout
- `run_sim.py`: single entry point to run demos or experiments from JSON configs.
- `configs/`: example configs (`demo.json`, `experiment.json`) and `experiment.schema.md` documenting fields.
- `modules/`: code for simulator (`simulator/runner.py`), controller, perception (legacy + LaneNet), HUD, and shared utils.
- `lanenet/`: LaneNet model code and weights (testing/benchmark only).
- `results/`: default output directory for trajectory CSVs when logging is enabled.

## Quick start
1) Prereqs: a Python env with CARLA Python API, pygame, OpenCV, TensorFlow (only for `lanenet`), scikit-learn, numpy.
2) Start a CARLA server (e.g., `CarlaUE4.sh -carla-port=2000`), default host/port `localhost:2000`.
3) Run with a config:
   - Demo (single run, HUD on, no logging):  
     ```bash
     python run_sim.py -c configs/demo.json
     ```
   - Experiment (multi-run, logging on):  
     ```bash
     python run_sim.py -c configs/experiment.json
     ```
4) To customize, edit a JSON per `configs/experiment.schema.md`. Key fields:
   - `experiment.mode`: `single` or `multi`
   - `environment.weather`: preset name from `configs/weather_presets.json`
   - `lkas.detector`: `legacy` (no TF) or `lanenet` (TF required)
   - `display.show_hud`: `true`/`false`
   - `logging.enabled`: `true`/`false`

### Output naming
When logging is enabled, trajectories are written to `results/` as:
```
<detector>_<map>_<weather>_<target_speed>_<street_light>_<vehicle_light>[_seedN]_trajectory.csv
```
`_seedN` is included only when a seed is supplied.

---

## LaneNet Model Notice
This repository includes the **LaneNet model** code, copied from  
[MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection).  

- LaneNet is **only used for testing purposes** in our experiments.  
- It is **not the backbone of our lane detection algorithm** within SimLKAS.  
- Full credit for LaneNet goes to the original authors. Please see their repository for details and licensing.

---

## Citation
If you use this framework in your research, please cite:

> Yang, T., Ding, Y., Li, K., Pan, Y., Qin, R., Yin, Z., & Hu, X. (2025).  
> SimLKAS: a simulation-based framework for verification and validation of lane keeping assistance systems.  
> *Journal of Intelligent Transportation Systems,* 1-25.  
> [https://doi.org/10.1080/15472450.2025.2559224](https://doi.org/10.1080/15472450.2025.2559224)
