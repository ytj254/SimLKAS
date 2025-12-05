import json
import os
from pathlib import Path
import carla
import numpy as np
import cv2

from modules.controller import VehicleController
from modules.hud import VehicleHUD, set_birdseye_map
from modules.perception import LegacyLaneDetector
from modules.utils import record_vehicle_data, save_to_csv, render_lane_overlay


REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_PRESETS_PATH = REPO_ROOT / "configs" / "weather_presets.json"


def apply_weather_preset(world, preset_name: str):
    if not WEATHER_PRESETS_PATH.exists():
        print(f"Weather preset file missing: {WEATHER_PRESETS_PATH}")
        return

    with open(WEATHER_PRESETS_PATH, 'r') as file:
        weather_presets = json.load(file)

    preset_values = weather_presets.get(preset_name)
    if not preset_values:
        print(f"Preset '{preset_name}' not found.")
        return

    weather = carla.WeatherParameters(
        cloudiness=preset_values[0],
        precipitation=preset_values[1],
        precipitation_deposits=preset_values[2],
        wind_intensity=preset_values[3],
        sun_azimuth_angle=preset_values[4],
        sun_altitude_angle=preset_values[5],
        fog_density=preset_values[6],
        fog_distance=preset_values[7],
        fog_falloff=preset_values[8],
        wetness=preset_values[9],
        scattering_intensity=preset_values[10],
        mie_scattering_scale=preset_values[11],
        rayleigh_scattering_scale=preset_values[12],
        dust_storm=preset_values[13]
    )
    world.set_weather(weather)


def build_detector(detector_name: str):
    if detector_name == "legacy":
        print("Using Legacy Lane Detector")
        return LegacyLaneDetector()
    elif detector_name == "lanenet":
        # Lazy import to avoid loading TensorFlow unless needed
        from modules.perception.lanenet_wrapper import build_lanenet_detector
        print("Using LaneNet Detector")    
        return build_lanenet_detector()
    else:
        raise ValueError(f"Unknown detector type: {detector_name}")


def _default_distance_for_map(map_name: str):
    if map_name == "Town04":
        return 1000.0
    return 350.0


def run_single_simulation(config: dict, seed=None):
    simulation_cfg = config.get("simulation", {})
    env_cfg = config.get("environment", {})
    lkas_cfg = config.get("lkas", {})
    display_cfg = config.get("display", {})
    logging_cfg = config.get("logging", {})

    map_name = simulation_cfg.get("map", "Town04")
    target_distance = simulation_cfg.get("distance_meters") or _default_distance_for_map(map_name)
    target_speed = env_cfg.get("target_speed", 80)
    weather = env_cfg.get("weather", "Default")
    street_light = env_cfg.get("street_light")
    veh_light = env_cfg.get("vehicle_light")
    detector_name = lkas_cfg.get("detector", "legacy")
    show_hud = display_cfg.get("show_hud", True)
    logging_enabled = logging_cfg.get("enabled", True)

    actor_list = []
    latest_frame = {"frame": None}

    if seed is not None:
        np.random.seed(seed)

    def image_callback(image):
        frame_np = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
        latest_frame["frame"] = rgb_image

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    client.load_world(map_name)
    world = client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 1 / 60
    world.apply_settings(settings)

    apply_weather_preset(world, weather)
    set_birdseye_map(world)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    if map_name == 'Town04':
        spawn_point = world.get_map().get_spawn_points()[38]
    elif map_name == 'Town10HD':
        spawn_point = world.get_map().get_spawn_points()[141]
    else:
        spawn_point = world.get_map().get_spawn_points()[197]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)

    time_of_day = 'Daytime'
    if 'Night' in weather:
        time_of_day = 'Night'
        if street_light == 'off':
            light_manager = world.get_lightmanager()
            street_lights = light_manager.get_all_lights(carla.LightGroup.Street)
            light_manager.turn_off(street_lights)
        if veh_light == 'LowBeam':
            vehicle.set_light_state(carla.VehicleLightState.LowBeam)
        elif veh_light == 'HighBeam':
            vehicle.set_light_state(carla.VehicleLightState.HighBeam)

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", str(640))
    cam_bp.set_attribute("image_size_y", str(480))
    cam_bp.set_attribute("fov", str(90))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2))
    ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    actor_list.append(ego_cam)
    ego_cam.listen(image_callback)

    detector = build_detector(detector_name)
    controller = VehicleController(vehicle, method='pid')
    vehicle_hud = VehicleHUD(vehicle, camera_view='third_person', video_file='veh_sim.mp4') if show_hud else None

    trajectory_data = []
    last_position = vehicle.get_location()
    total_distance_traveled = 0.0

    try:
        if vehicle_hud:
            vehicle_hud.start()

        while True:
            lateral_deviation = record_vehicle_data(
                vehicle, world, trajectory_data,
                target_speed,
                weather,
                time_of_day,
                street_light,
                veh_light
            )

            current_position = vehicle.get_location()
            increment_distance = np.sqrt(
                (current_position.x - last_position.x) ** 2 +
                (current_position.y - last_position.y) ** 2
            )
            total_distance_traveled += increment_distance
            last_position = current_position

            if total_distance_traveled >= target_distance:
                print(f"Target distance of {target_distance / 1000:.2f} km reached. Simulation stopped.")
                break

            if abs(lateral_deviation) >= 1.5:
                print(f"Lateral deviation of {lateral_deviation:.2f} meters exceeded the threshold. Simulation stopped.")
                print(f"Total of {total_distance_traveled / 1000:.2f} km traveled")
                break

            frame = latest_frame["frame"]
            if frame is not None:
                line_parameters = detector.detect_lane(frame)
                control = controller.run_step(target_speed, line_parameters)
                vehicle.apply_control(control)

                if vehicle_hud:
                    overlay_frame = render_lane_overlay(frame, line_parameters)
                    vehicle_hud.update_lane_overlay(overlay_frame)
            elif vehicle_hud:
                vehicle_hud.update_lane_overlay(None)

            if vehicle_hud:
                vehicle_hud.update()
            world.tick()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if logging_enabled:
            results_dir = REPO_ROOT / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            filename_parts = [
                detector_name,
                map_name,
                weather,
                str(target_speed),
                str(street_light),
                str(veh_light),
            ]
            if seed is not None:
                filename_parts.append(f"seed{seed}")
            trajectory_file = results_dir / f"{'_'.join(filename_parts)}_trajectory.csv"
            save_to_csv(trajectory_data, trajectory_file, headers=[
                "time", "x", "y", "deviation", "curved", "target_speed",
                "weather", "time_of_day", "street_light", "veh_light"
            ])

        for actor in actor_list:
            actor.destroy()
        if vehicle_hud:
            vehicle_hud.destroy()
        cv2.destroyAllWindows()


def run_from_config(config: dict):
    experiment_cfg = config.get("experiment", {})
    mode = experiment_cfg.get("mode", "single")

    if mode == "single":
        run_single_simulation(config)
    elif mode == "multi":
        conditions = experiment_cfg.get("conditions", [])
        seeds = experiment_cfg.get("seeds")
        num_runs = experiment_cfg.get("num_runs", 1)
        if seeds:
            seed_iterable = seeds
        else:
            seed_iterable = [None] * num_runs
        if not conditions:
            conditions = [{}]
        for seed in seed_iterable:
            for condition in conditions:
                merged = _merge_condition(config, condition)
                run_single_simulation(merged, seed)
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")


def _merge_condition(base_config: dict, condition: dict):
    merged = json.loads(json.dumps(base_config))  # deep copy via json
    env_override = condition.get("environment", {})
    lkas_override = condition.get("lkas", {})

    merged.setdefault("environment", {}).update(env_override)
    merged.setdefault("lkas", {}).update(lkas_override)
    return merged
