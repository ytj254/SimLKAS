import carla
import numpy as np
import cv2
from simple_pid import PID
import time

# Initialize PID controller
pid = PID(1.0, 0.1, 0.05, setpoint=0)  # Tune the PID gains based on performance


# Function to process camera image and detect lane center
def detect_lane_lines(image):
    # Convert CARLA image to NumPy array
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    img = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    left_lines = []
    right_lines = []
    height, width = img.shape[:2]
    image_center = width // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            if slope < -0.5:  # Left lane
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:  # Right lane
                right_lines.append((x1, y1, x2, y2))

    # Average lines to determine lane center
    def average_line(lines):
        x_coords = [x1 for x1, _, x2, _ in lines] + [x2 for _, _, x2, _ in lines]
        return sum(x_coords) / len(x_coords) if lines else None

    left_lane_center = average_line(left_lines)
    right_lane_center = average_line(right_lines)

    # Calculate lane center
    lane_center = (left_lane_center + right_lane_center) / 2 if left_lane_center and right_lane_center else None

    # Visualize for debugging
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Lane Detection", img)
    cv2.waitKey(1)

    return lane_center, image_center


# Function to calculate steering angle
def calculate_steering(lane_center, image_center):
    if lane_center is not None:
        offset = image_center - lane_center
        return pid(offset / image_center)  # Normalize the offset
    return 0  # Default to straight driving if lane is not detected


# Function to control the vehicle
def control_vehicle(vehicle, steering_angle):
    control = carla.VehicleControl()
    control.throttle = 0.5  # Fixed throttle for simplicity
    control.steer = np.clip(steering_angle, -1.0, 1.0)  # Ensure steering is within [-1, 1]
    vehicle.apply_control(control)


# Main function
def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    # Load a new map
    client.load_world('Town03')
    world = client.get_world()

    # Spawn the vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    spectator = world.get_spectator()
    vehicle_transform = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(vehicle_transform.location + carla.Location(z=10), vehicle_transform.rotation))

    # Attach camera
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x", str(640))
    cam_bp.set_attribute("image_size_y", str(480))
    cam_bp.set_attribute("fov", str(105))
    cam_location = carla.Location(2, 0, 1)
    cam_rotation = carla.Rotation(0, 0, 0)
    cam_transform = carla.Transform(cam_location, cam_rotation)
    ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle,
                                attachment_type=carla.AttachmentType.Rigid)

    # Set up camera listener
    lane_center = None
    image_center = None

    def process_image(image):
        # Convert the raw image data to a numpy array
        array = np.array(image.raw_data)
        array = array.reshape(image.height, image.width, 4)  # RGBA format
        array = array[:, :, :3]  # Convert to RGB

        # Display the image using OpenCV
        cv2.imshow('CARLA Camera', array)
        cv2.waitKey(1)

    def process_camera_image(image):
        nonlocal lane_center, image_center
        lane_center, image_center = detect_lane_lines(image)
        print(lane_center, image_center)

    # Control loop
    try:
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        ego_cam.listen(lambda image: process_image(image))
        time.sleep(5)

        # while True:
        #     if lane_center is not None and image_center is not None:
        #         steering_angle = calculate_steering(lane_center, image_center)
        #         control_vehicle(vehicle, steering_angle)
        #
        #
        #     time.sleep(0.05)  # Simulate real-time control loop
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Cleanup
        ego_cam.stop()
        ego_cam.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
