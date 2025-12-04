import carla
import numpy as np
import cv2
import json

from sympy.codegen.ast import continue_

from lane_detect import LegacyLaneDetector
from Controller import VehicleController
from lanenet_detect import LaneDetector
from utils import record_vehicle_data, save_to_csv
from vehicle_hud import VehicleHUD, set_birdseye_map
import itertools

# Define latest_frame globally at the module level
latest_frame = None

def image_callback(image):
    """
    Callback function to process and store the latest camera frame.
    """
    global latest_frame

    # Convert CARLA raw image to NumPy array
    frame_np = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

    # Store the latest frame
    latest_frame = rgb_image

# Function to apply a preset
def apply_weather_preset(world, preset_name):
    # Load presets from JSON file
    with open('weather_presets.json', 'r') as file:
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

def sim_run(lane_detector='legacy', target_speed=80, weather='Default', street_light=None, veh_light=None, map='Town04', log=True, i=1, veh_hud=True):
    actor_list = []

    # Control loop
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    # Load a new map
    client.load_world(map)
    world = client.get_world()
    # Set synchronous mode
    settings = world.get_settings()
    # settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / 60  # 50 FPS
    world.apply_settings(settings)
    # print(settings)

    # Set the weather
    apply_weather_preset(world, weather)

    set_birdseye_map(world)

    # Spawn the vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    if map == 'Town04':
        spawn_point = world.get_map().get_spawn_points()[38] # backup 367
    elif map == 'Town10HD':
        spawn_point = world.get_map().get_spawn_points()[141]
    else: # Town03
        spawn_point = world.get_map().get_spawn_points()[197]  # 197, 196

    # print(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)

    # Set the lights
    if 'Night' in weather:
        time_of_day = 'Night'
        if street_light == 'off':
            print('Night, turn off street lights')
            light_manager = world.get_lightmanager()
            street_lights = light_manager.get_all_lights(carla.LightGroup.Street)
            light_manager.turn_off(street_lights)

        # Control vehicle lights
        if veh_light == 'LowBeam':
            print('Turn on low beam.')
            vehicle.set_light_state(carla.VehicleLightState.LowBeam)
        elif veh_light == 'HighBeam':
            print('Turn on high beam.')
            vehicle.set_light_state(carla.VehicleLightState.HighBeam)
        else:
            pass
    else:
        time_of_day = 'Daytime'

    # Attach camera
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", str(640))
    cam_bp.set_attribute("image_size_y", str(480))
    cam_bp.set_attribute("fov", str(90))
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2))
    ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    actor_list.append(ego_cam)

    # vehicle.apply_control(carla.VehicleControl(throttle=1.0))
    # vehicle.set_autopilot(True)
    ego_cam.listen(lambda image: image_callback(image))

    if lane_detector == 'legacy':
        print("Using LegacyLaneDetector.")
        detector = LegacyLaneDetector()
    else:
        print("Using LaneNet detector.")
        from lanenet.laneNet_class import LaneNet
        model = LaneNet()
        detector = LaneDetector(model)

    controller = VehicleController(vehicle, method='pid')
    if veh_hud:
        vehicle_hud = VehicleHUD(vehicle, camera_view='third_person', video_file='veh_sim.mp4')
    # Initialize trajectory storage
    trajectory_data = []
    # Initialize variables for distance tracking
    last_position = vehicle.get_location()
    total_distance_traveled = 0.0  # Total distance in meters

    # Target distance in meters (Town04: 1 km; Town03: 350 m)
    if map == 'Town04':
        target_distance = 1000.0
    else:
        target_distance = 350.0
    try:

        # Start the HUD explicitly when needed
        if veh_hud:
            vehicle_hud.start()
        # vehicle_hud.start_recording()
        while True:
            # Record vehicle data
            lateral_deviation = record_vehicle_data(
                vehicle, world, trajectory_data,
                target_speed,
                weather,
                time_of_day,
                street_light,
                veh_light
            )

            # Track the vehicle's traveled distance
            current_position = vehicle.get_location()
            # host_waypoint = world.get_map().get_waypoint(current_position)
            # print(host_waypoint.lane_width)

            increment_distance = np.sqrt(
                (current_position.x - last_position.x) ** 2 +
                (current_position.y - last_position.y) ** 2
            )
            total_distance_traveled += increment_distance
            last_position = current_position  # Update last position
            # Check if the target distance is reached
            if total_distance_traveled >= target_distance:
                print(f"Target distance of {target_distance / 1000:.2f} km reached. Simulation stopped.")
                break

            if abs(lateral_deviation) >= 1.5:
                print(f"Lateral deviation of {lateral_deviation:.2f} meters exceeded the threshold. Simulation stopped.")
                print(f"Total of {total_distance_traveled / 1000:.2f} km traveled")
                break

            if latest_frame is not None:
                # target_speed = vehicle.get_speed_limit()
                # print(target_speed)
                line_parameters = detector.detect_lane(latest_frame)
                # print(line_parameters)
                control = controller.run_step(target_speed, line_parameters)
                vehicle.apply_control(control)
            if veh_hud:
                vehicle_hud.update()
            world.tick()
    except KeyboardInterrupt:
            print("Exiting...")
    finally:
        # Cleanup
        print('Cleaning up')

        # Save the recorded data to CSV for later use
        if log:
            trajectory_file = f"results/town04/{lane_detector}_{map}_{weather}_{target_speed}_{street_light}_{veh_light}_vehicle_trajectory_{i}.csv"
            save_to_csv(trajectory_data, trajectory_file, headers=["time", "x", "y", "deviation", "curved", "target_speed", "weather", "time_of_day", "street_light", "veh_light"])

        # Clean up the actors and resources
        for actor in actor_list:
            actor.destroy()
        # vehicle_hud.stop_recording()
        if veh_hud:
            vehicle_hud.destroy()
        cv2.destroyAllWindows()

def experiments_run(detector, target_speed, map):
    # Define the scenarios
    scenarios = (
        ('ClearNoon', None, None),
        ('SmallRainNoon', None, None),
        ('HardRainNoon', None, None),
        ('MidFogNoon', None, None),
        ('HeavyFogNoon', None, None),
        ('ClearNight', 'on', 'LowBeam'),
        ('ClearNight', 'on', 'HighBeam'),
        ('ClearNight', 'off', 'LowBeam'),
        ('ClearNight', 'off', 'HighBeam'),
    )

    # Loop through each scenario
    for i in range(3):
        for scenario in scenarios:
            weather, street_light, veh_light = scenario
            print(f'Current running scenario: {scenario}, {i+1}')
            sim_run(
                lane_detector=detector, # legacy or lanenet
                target_speed=target_speed,
                weather=weather,
                street_light=street_light,
                veh_light=veh_light,
                map=map,
                log=True,
                i=i+1,
            )
    print(f'Experiment of {target_speed} is completed.')

if __name__ == '__main__':
    # speeds = [30, 40, 50, 60]
    # for speed in speeds:
    settings = {
        'lane_detector': 'legacy',
        'target_speed': 90,
        'weather': 'ClearNoon',
        'street_light': None, # on, off
        'veh_light': None, # LowBeam, HighBeam
        'map': 'Town04',
        }
        # for i in range(3):
        #     sim_run(
        #         lane_detector = settings['lane_detector'],
        #         target_speed = settings['target_speed'],
        #         weather = settings['weather'],
        #         street_light = settings['street_light'],
        #         veh_light = settings['veh_light'],
        #         map = settings['map'],
        #         log=True,
        #         i=i+1,
        #         veh_hud=False,
        #     )

    sim_run(
        lane_detector=settings['lane_detector'],
        target_speed=settings['target_speed'],
        weather=settings['weather'],
        street_light=settings['street_light'],
        veh_light=settings['veh_light'],
        map=settings['map'],
        log=False,
        i=3,
        veh_hud=True,
    )
    # experiments_run(detector='legacy', target_speed=30, map='Town04')
