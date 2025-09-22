import carla
import pygame
import numpy as np
import cv2


class VehicleHUD:
    def __init__(self, vehicle, camera_view='first_person', width=800, height=600, video_file=None):
        self.vehicle = vehicle
        self.camera = None
        self.image = None
        self.world = vehicle.get_world()
        self.width = width
        self.height = height
        self.camera_view = camera_view

        # Flags to manage Pygame initialization
        self.pygame_initialized = False

        # Video recording attributes
        self.recording = False
        self.video_writer = None
        self.video_file = video_file
        self.fps = 30

        # Default camera settings
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(width))
        self.camera_bp.set_attribute('image_size_y', str(height))
        self.camera_bp.set_attribute('fov', '90')

        # Default camera view
        self.set_camera_view(camera_view)

    def _process_image(self, image):
        """Convert CARLA raw image to a format compatible with Pygame and record video."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
        self.image = array[:, :, :3][:, :, ::-1]  # Convert to RGB for Pygame display

    def _add_vehicle_data_to_frame(self, frame, vehicle_data):
        """Overlay vehicle data on the frame."""
        overlay_frame = frame.copy()
        speed, throttle, steering, brake = vehicle_data

        # Overlay text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        color = (255, 255, 255)
        thickness = 2

        cv2.putText(overlay_frame, f"Speed: {speed:.2f} km/h", (10, 30), font, scale, color, thickness)
        cv2.putText(overlay_frame, f"Throttle: {throttle:.2f}", (10, 60), font, scale, color, thickness)
        cv2.putText(overlay_frame, f"Steering: {steering:.2f}", (10, 90), font, scale, color, thickness)
        cv2.putText(overlay_frame, f"Brake: {brake:.2f}", (10, 120), font, scale, color, thickness)

        return overlay_frame

    def get_vehicle_data(self):
        """Retrieve vehicle control and speed information."""
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # Convert m/s to km/h

        control = self.vehicle.get_control()
        throttle = control.throttle
        steering = control.steer
        brake = control.brake

        return speed, throttle, steering, brake

    def set_camera_view(self, view_type):
        """
        Change the camera's position and orientation dynamically.

        :param view_type: The desired view type ('first_person', 'third_person', 'bird_eye').
        """
        # Destroy the existing camera
        if self.camera:
            self.camera.destroy()

        # Set camera based on the view type
        if view_type == "first_person":
            # Camera in front of the vehicle
            self.camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        elif view_type == "third_person":
            # Camera behind and above the vehicle
            self.camera_transform = carla.Transform(
                location=carla.Location(x=-5.0, z=3.0),  # Behind and above the vehicle
                rotation=carla.Rotation(pitch=-15.0)     # Tilt downward
            )

        elif view_type == "bird_eye":
            # Overhead view
            self.camera_transform = carla.Transform(
                location=carla.Location(x=0, z=20.0),    # Directly above the vehicle
                rotation=carla.Rotation(pitch=-90.0)     # Looking straight down
            )

        else:
            raise ValueError(f"Unknown view type: {view_type}")

        # Spawn the new camera
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_image(image))

    def start_recording(self):
        """Start recording video."""
        if self.video_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_file, fourcc, self.fps, (self.width, self.height))
            self.recording = True
            print(f"Recording started: {self.video_file}")

    def stop_recording(self):
        """Stop recording video."""
        if self.recording:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped.")

    def start(self):
        """Initialize Pygame resources, video writer, and display the HUD window."""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Vehicle HUD")
            self.font = pygame.font.Font(None, 36)
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True

    def update(self):
        """Update the display with the latest camera feed and vehicle data."""
        # Handle Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt  # Graceful exit
        # Get vehicle data once per frame
        vehicle_data = self.get_vehicle_data()

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Display camera feed
        if self.image is not None:
            # Add vehicle data overlay for Pygame display
            overlay_frame = self._add_vehicle_data_to_frame(self.image, vehicle_data)

            # Convert overlay frame to Pygame surface
            surface = pygame.surfarray.make_surface(overlay_frame.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))

            # Write to video file if recording
            if self.recording and self.video_writer:
                self.video_writer.write(overlay_frame[:, :, ::-1])  # Convert back to BGR for OpenCV

        pygame.display.flip()
        self.clock.tick(self.fps)

    def destroy(self):
        """Cleanup the camera resources, video writer, and Pygame."""
        if self.camera:
            self.camera.destroy()
        if self.recording:
            self.stop_recording()
        pygame.quit()


def set_birdseye_map(world):
    spectator = world.get_spectator()

    # Get the bounds of the map
    map_bounds = world.get_map().get_spawn_points()
    x_coords = [point.location.x for point in map_bounds]
    y_coords = [point.location.y for point in map_bounds]

    # Calculate the center of the map
    center_x = (max(x_coords) + min(x_coords)) / 2
    center_y = (max(y_coords) + min(y_coords)) / 2

    # Set the spectator's transform for bird's-eye view
    birdseye_height = 300  # Height above the map in meters
    spectator.set_transform(carla.Transform(
        carla.Location(x=center_x, y=center_y, z=birdseye_height),  # Position above the map
        carla.Rotation(pitch=-90, yaw=0, roll=0)  # Looking straight down
    ))

def draw_spawn_points(world):
    # Retrieve spawn points
    map = world.get_map()
    spawn_points = map.get_spawn_points()

    # Draw spawn points on the map
    for i, spawn_point in enumerate(spawn_points):
        location = spawn_point.location
        rotation = spawn_point.rotation

        # Draw an arrow at the spawn point
        world.debug.draw_arrow(
            location,  # Start of the arrow
            location + carla.Location(x=2 * rotation.get_forward_vector().x,
                                      y=2 * rotation.get_forward_vector().y,
                                      z=2 * rotation.get_forward_vector().z),  # End of the arrow
            thickness=0.1, arrow_size=0.5, color=carla.Color(255, 0, 0), life_time=0.0
        )

        # Optionally, draw text showing the spawn point index
        world.debug.draw_string(
            location, f"Spawn {i}", draw_shadow=True, color=carla.Color(0, 255, 0), life_time=1000.0
        )

    print(f"Visualized {len(spawn_points)} spawn points.")

if __name__ == '__main__':
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world('Town04')
    world = client.get_world()
    print(world.get_weather())
    # cloudiness = 10.000000,
    # precipitation = 0.000000,
    # precipitation_deposits = 10.000000,
    # wind_intensity = 30.000000,
    # sun_azimuth_angle = 150.000000,
    # sun_altitude_angle = 60.000000,
    # fog_density = 40.000000,
    # fog_distance = 60.000000,
    # fog_falloff = 2.000000,
    # wetness = 30.000000,
    # scattering_intensity = 1.000000,
    # mie_scattering_scale = 0.030000,
    # rayleigh_scattering_scale = 0.033100,
    # dust_storm = 0.000000
    set_birdseye_map(world)
    draw_spawn_points(world)