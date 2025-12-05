import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Define the region of interest (source points)
# These points should form a trapezoid around the lane in the input image
src_points = np.float32([
    [275, 290],  # Top-left
    [365, 290],  # Top-right
    [110, 480],  # Bottom-left
    [530, 480]  # Bottom-right
])

# Define the rectangle for the bird's-eye view (destination points)
# These points define where the trapezoid will map to in the bird's-eye view
dst_points = np.float32([
    [110, 0],  # Top-left
    [530, 0],  # Top-right
    [110, 480],  # Bottom-left
    [530, 480]  # Bottom-right
])

# Define the image shape
img_height = 480
img_width = 640

# Define the weather presets


def show_image(title, frame):
    cv2.imshow(title, frame)
    cv2.waitKey(0)

def show_video(title, frame):
    cv2.imshow(title, frame)
    cv2.waitKey(1)

def plt_image(frame):
    plt.imshow(frame)
    plt.show()

def warp_perspective_to_birdseye(image):
    """
    Transform an image into a bird’s-eye view using perspective transformation.

    Parameters:
    - image: Input image (e.g., from CARLA camera).
    - src_points: Four points (clockwise or counterclockwise) in the input image to define the region of interest (trapezoid).
    - dst_points: Four points in the output image to define the rectangular bird’s-eye view.
    - output_size: Tuple (width, height) for the output image size.

    Returns:
    - Transformed bird’s-eye view image.
    """
    # Define the output size (width, height)
    output_size = (img_width, img_height)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image to the bird’s-eye view
    birdseye_image = cv2.warpPerspective(image, M, output_size)
    # show_image('birdseye', birdseye_image)
    return birdseye_image, M

def render_lane_overlay(
        frame: np.ndarray,
        lane_lines: Optional[Dict[str, Tuple]],
        alpha: float = 0.6,
):
    """
    Create an overlay of perceived lane markings on the raw frame.

    The lane lines are defined in the bird's-eye-view coordinate frame and are
    projected back into the camera view for display.
    """
    if lane_lines is None or frame is None:
        return None

    birdseye_image, transform_matrix = warp_perspective_to_birdseye(frame)
    if transform_matrix is None:
        return None

    overlay = np.zeros_like(birdseye_image)
    y_bottom = img_height - 1
    y_top = 0

    color_map = {
        "left": (0, 0, 255),   # Red for left lane
        "right": (0, 255, 0),  # Green for right lane
    }

    for side, params in lane_lines.items():
        if params is None:
            continue
        # Some detectors return (None, None) when nothing is found; skip those
        if len(params) < 4 or any(p is None for p in params):
            continue
        vx, vy, x0, y0 = params
        if vx == 0:
            continue  # Avoid vertical slope division
        slope = vy / vx
        intercept = y0 - slope * x0

        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        x_bottom = max(0, min(img_width - 1, x_bottom))
        x_top = max(0, min(img_width - 1, x_top))

        color = color_map.get(side, (255, 255, 0))
        cv2.line(overlay, (x_bottom, y_bottom), (x_top, y_top), color, 6)

    try:
        minv = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError:
        return None

    # Project the lane overlay back onto the camera frame
    overlay_camera = cv2.warpPerspective(overlay, minv, (frame.shape[1], frame.shape[0]))
    blended = cv2.addWeighted(frame, 1.0, overlay_camera, alpha, 0)
    return blended

def check_src_points(image):
    image_copy = image.copy()
    plt_image(image_copy)

    for i in range(len(src_points)):
        pt1 = tuple(src_points[i].astype(int))
        pt2 = tuple(src_points[(i + 1) % len(src_points)].astype(int))
        cv2.line(image_copy, pt1, pt2, (0, 255, 0), 2)  # Green trapezoid
    plt_image(image_copy)

def record_vehicle_data(
        vehicle, world, trajectory_data,
        target_speed,
        weather,
        time_of_day,
        street_light,
        veh_light,
        waypoint_distance=2
):
    """
       Generate or update the centerline and record vehicle position with lateral deviation.

       :param vehicle: CARLA vehicle actor.
       :param world: CARLA world instance.
       :param trajectory_data: List to store (time, x, y, deviation) tuples.
       :param target_speed: scenario setting details.
       :param veh_light:
       :param street_light:
       :param time_of_day:
       :param weather:
       :param waypoint_distance: Distance between consecutive waypoints in meters.
       """
    # Get the vehicle's current location
    vehicle_location = vehicle.get_location()
    vehicle_x = vehicle_location.x
    vehicle_y = vehicle_location.y

    # Get the current waypoint
    host_waypoint = world.get_map().get_waypoint(vehicle_location)

    # Get previous, current, and next waypoints
    try:
        next_waypoint = host_waypoint.next(waypoint_distance)[0]
    except IndexError:
        return float('nan')  # Return NaN if waypoints are unavailable

    # Identify the lane geometry
    curved = is_lane_curved(host_waypoint, next_waypoint)

    # Extract the positions of the two points
    host_pos = (host_waypoint.transform.location.x, host_waypoint.transform.location.y)
    next_pos = (next_waypoint.transform.location.x, next_waypoint.transform.location.y)

    # Calculate the vector of the centerline segment
    line_vector = np.array(next_pos) - np.array(host_pos)

    # Calculate the vector from the closest point to the vehicle
    vehicle_vector = np.array([vehicle_x, vehicle_y]) - np.array(host_pos)

    # Calculate the perpendicular distance (lateral deviation)
    lateral_deviation = np.linalg.norm(
        np.cross(line_vector, vehicle_vector) / np.linalg.norm(line_vector)
    )

    # Determine the side of the deviation (left or right)
    direction = np.sign(np.cross(line_vector, vehicle_vector))
    lateral_deviation *= direction
    # print(f'Old lateral deviation:', lateral_deviation)

    # Record the data
    current_time = world.get_snapshot().timestamp.elapsed_seconds
    trajectory_data.append((
        current_time,
        vehicle_x,
        vehicle_y,
        lateral_deviation,
        curved,
        target_speed,
        weather,
        time_of_day,
        street_light,
        veh_light
    ))
    return lateral_deviation

def is_lane_curved(host_waypoint, next_waypoint,curvature_threshold=0.2):
    """
    Determine whether the current lane is straight or curved.

    :param host_waypoint: The nearest waypoint.
    :param next_waypoint: The next waypoint.
    :param curvature_threshold: Threshold for angular change (degrees) to identify curvature.
    :return: True if the lane is curved, False if the lane is straight.
    """
    # Calculate angular change between the current and next waypoint
    yaw_host = host_waypoint.transform.rotation.yaw
    yaw_next = next_waypoint.transform.rotation.yaw
    angular_change = abs(yaw_next - yaw_host)
    # print(f"Angular Change: {angular_change} degrees")

    # Normalize the angular change to the range [0, 180]
    angular_change = min(angular_change, 360 - angular_change)
    # print(f"Angular Change after normalizing: {angular_change} degrees")

    # Determine if the lane is curved
    return angular_change > curvature_threshold

def save_to_csv(data, filename, headers):
    """
    Save data to a CSV file.

    :param data: List of tuples representing the data to save.
    :param filename: Path to the output CSV file.
    :param headers: List of column names.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers
        writer.writerows(data)  # Write data
    print(f"Data saved to {filename}")


if __name__ == '__main__':
    file = 'images/town03_raw image_screenshot_31.12.2024.png'
    raw_frame = cv2.imread(file)
    frame = np.copy(raw_frame)
    check_src_points(frame)
