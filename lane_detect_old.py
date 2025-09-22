import cv2
import numpy as np
import logging
import math
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import datetime
import sys
from utils import *

def detect_lane(frame):
    logging.debug('detecting lane lines...')
    # show_vedio('raw frame', frame)
    canny_img = canny(frame)
    # show_vedio('canny_img', canny_img)
    # show_image('canny_img', canny_img)

    # cropped_img= region_of_interest(canny_img)
    # show_image('image cropped', cropped_img)
    birdseye_image, _ = warp_perspective_to_birdseye(frame)

    birdseye_canny, _ = warp_perspective_to_birdseye(canny_img)

    # show_image('bird', frame)
    # show_vedio('birdseye', frame)

    lane_lines_image = identify_lane_marking_lines(birdseye_canny)
    # show_image('lane marking lines', lane_lines_image)


    lane_lines = identify_lane_marking_params(birdseye_canny)
    visualized_image = draw_lines(birdseye_image, lane_lines)

    # show_image('lane marking lines', visualized_image)
    show_vedio('lane marking lines', visualized_image)


    # line_segments = detect_line_segments(frame)
    # print(line_segments)
    # line_segment_image = display_lines(frame, line_segments)
    # show_image("line segments", line_segment_image)
    # show_vedio("line segments", line_segment_image)

    # lane_lines = identify_lane_marking(line_segments, frame)
    # print(lane_lines)
    # lane_lines_image = display_lines(frame, lane_lines)
    # show_vedio("lane_lines_image", lane_lines_image)

    return lane_lines

def canny(frame):
    # print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # show_image('blur', blur)
    canny_img = cv2.Canny(blur, 30, 100)
    # print(canny_img.shape)
    return canny_img

def region_of_interest(canny_img):
    height, width = canny_img.shape
    # print(canny_img.shape)
    # only focus bottom half of the screen
    polygon = np.array([
    [(0, height), (width, height), (width, height / 2), (0, height / 2)]  # Select the min area
    ], dtype=np.int32)

    mask = np.zeros_like(canny_img)
    cv2.fillPoly(mask, polygon, (255, 255, 255))
    # show_image('mask', mask)
    masked_image = cv2.bitwise_and(canny_img, mask)
    return masked_image


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]),
                                    minLineLength=8, maxLineGap=4)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('detected line_segment:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments

def fit_lane(points):
    """
    Fit a lane line using RANSAC given slope and intercept points.
    """
    if len(points) < 2:
        return None, None

    slopes = np.array([p[0] for p in points]).reshape(-1, 1)
    intercepts = np.array([p[1] for p in points])

    try:
        ransac = RANSACRegressor()
        ransac.fit(slopes, intercepts)

        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        return slope, intercept
    except Exception as e:
        print(f"Error in RANSAC fitting: {e}")
        return None, None

def identify_lane_marking(lines, image):
    """
    Process the lines detected by cv2.HoughLinesP to separate left and right lanes
    and return the left and right lane lines in the same format as HoughLinesP.

    Parameters:
        lines: Output of cv2.HoughLinesP (array of lines as (x1, y1, x2, y2)).
        image: Input image where the lanes are detected.

    Returns:
        left_lines: List of arrays [(x1, y1, x2, y2), ...] for the left lane.
        right_lines: List of arrays [(x1, y1, x2, y2), ...] for the right lane.
    """
    min_len = 0
    if lines is None or len(lines) == 0:
        print("No lines detected.")
        return [], []

    x_middle = img_width / 2  # Find the middle of the image

    left_candidates = []
    right_candidates = []

    # Iterate over lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            # # Calculate the length of the line
            # line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # if line_length < min_len:  # Exclude short lines
            #     continue

            # Calculate slope and intercept
            if x2 == x1:  # Skip vertical lines to avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)

            if slope == 0:  # Skip horizontal lines
                continue

            intercept = y1 - slope * x1  # y-intercept of the line

            # Calculate the x-intercept at the bottom of the image
            bottom_x = (img_height - intercept) / slope

            # Classify line as left or right based on its slope and bottom_x position
            if slope < 0 and bottom_x < x_middle:
                left_candidates.append((abs(bottom_x - x_middle), [x1, y1, x2, y2]))
            elif slope > 0 and bottom_x > x_middle:
                right_candidates.append((abs(bottom_x - x_middle), [x1, y1, x2, y2]))

    # Select the closest line to the middle for both left and right
    left_line = min(left_candidates, key=lambda x: x[0])[1] if left_candidates else None
    right_line = min(right_candidates, key=lambda x: x[0])[1] if right_candidates else None

    return left_line, right_line

def compute_steering_angle(left_line, right_line, frame):
    """
    Calculate the steering angle based on the left and right lane lines.

    Parameters:
        left_line: Closest line on the left ([x1, y1, x2, y2]).
        right_line: Closest line on the right ([x1, y1, x2, y2]).
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        steering_angle: Steering angle in degrees.
    """
    frame_height, frame_width, _ = frame.shape
    if left_line is None or right_line is None:
        print("Both left and right lines are required to calculate the steering angle.")
        return None

    # Calculate the bottom points of the left and right lines
    left_slope = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
    left_intercept = left_line[1] - left_slope * left_line[0]
    left_bottom_x = (frame_height - left_intercept) / left_slope

    right_slope = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
    right_intercept = right_line[1] - right_slope * right_line[0]
    right_bottom_x = (frame_height - right_intercept) / right_slope

    # Calculate the lane center
    lane_center_x = (left_bottom_x + right_bottom_x) / 2

    # Calculate the vehicle's center
    vehicle_center_x = frame_width / 2

    # Calculate the deviation
    deviation = lane_center_x - vehicle_center_x

    # Calculate the steering angle in radians
    steering_angle_rad = math.atan(deviation / frame_height)

    # Convert the angle to degrees
    steering_angle_deg = math.degrees(steering_angle_rad)

    return steering_angle_deg


############################
# Utility Functions
############################
def display_lines(frame, lines):
    # Convert Canny image to color
    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    line_image = np.zeros_like(color_frame)  # Blank image for lines
    if lines is not None:
        for line in lines:
            # for x1, y1, x2, y2 in line:
            if line is not None and len(line) == 4:  # Check for valid line format
                x1, y1, x2, y2 = line
                # Draw the line on the blank image
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines

    # Overlay the detected lines on the original image
    combined_image = cv2.addWeighted(color_frame, 0.8, line_image, 1, 0)
            # if line is not None:
            #     print(line)
            #     x1, y1, x2, y2 = line
            #     cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    # line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined_image

def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


############################
# Test Functions
############################
def test_photo(file):
    # land_follower = HandCodedLaneFollower()
    frame = cv2.imread(file)
    detect_lane(frame)
    # combo_image = land_follower.follow_lane(frame)
    # cv2.imshow('final', lane_lines_image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

def identify_lane_marking_lines(edge_image):
    """
    Identify lane marking lines from an image with small rectangular lane markings.

    :param edge_image: Binary edge-detected image.
    :return: Dictionary containing left and right lane parameters.
    """
    # Step 1: Detect contours
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Filter contours by size (to ignore noise)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Adjust threshold based on your specific image
            filtered_contours.append(contour)

    # Step 3: Get all points from the filtered contours
    points = []
    for contour in filtered_contours:
        for point in contour:
            points.append(point[0])  # Extract (x, y) coordinates

    points = np.array(points)  # Convert to NumPy array

    # Step 4: Fit lines using Hough Transform or polynomial fitting
    line_image = np.zeros_like(edge_image)

    if len(points) > 0:
        # Separate points into left and right based on x-coordinates
        height, width = edge_image.shape
        left_points = points[points[:, 0] < width // 2]
        right_points = points[points[:, 0] >= width // 2]

        def fit_and_draw(points, color):
            if len(points) > 0:
                # Fit a polynomial to the points
                poly_fit = np.polyfit(points[:, 1], points[:, 0], 1)  # x = m*y + c
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                x_min, x_max = np.polyval(poly_fit, [y_min, y_max])
                # Draw the line
                cv2.line(line_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

        # Fit and draw left lane
        fit_and_draw(left_points, 255)  # White line

        # Fit and draw right lane
        fit_and_draw(right_points, 255)  # White line

    return line_image

def visualize_lane_lines(image, lane_params):
    """
    Visualize detected lane lines on an image.

    :param image: Original or blank image.
    :param lane_params: Dictionary with "left" and "right" line parameters.
                        Each is a tuple (vx, vy, x0, y0).
    :return: Image with visualized lane lines.
    """
    output_image = image.copy()  # Create a copy to draw on
    height, width = image.shape[:2]

    # Helper function to draw a line based on parameters
    def draw_line(vx, vy, x0, y0, color, thickness=2):
        # Calculate two points far apart on the line
        x1 = int(x0 - vx * height)  # Point far up
        y1 = int(y0 - vy * height)
        x2 = int(x0 + vx * height)  # Point far down
        y2 = int(y0 + vy * height)
        cv2.line(output_image, (x1, y1), (x2, y2), color, thickness)

    # Draw left lane
    if lane_params["left"] is not None:
        vx, vy, x0, y0 = lane_params["left"]
        draw_line(vx, vy, x0, y0, (0, 255, 0))  # Green line for left lane

    # Draw right lane
    if lane_params["right"] is not None:
        vx, vy, x0, y0 = lane_params["right"]
        draw_line(vx, vy, x0, y0, (255, 0, 0))  # Blue line for right lane

    return output_image

def identify_lane_marking_params(edge_image):
    """
    Identify lane marking lines and compute line parameters.

    :param edge_image: Binary edge-detected image.
    :return: Dictionary with left and right lane line parameters.
    """
    # Step 1: Detect contours
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Filter contours by size (to ignore noise)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Adjust threshold based on your specific image
            filtered_contours.append(contour)

    # Step 3: Get all points from the filtered contours
    points = []
    for contour in filtered_contours:
        for point in contour:
            points.append(point[0])  # Extract (x, y) coordinates

    points = np.array(points)  # Convert to NumPy array

    # Step 4: Fit lines and compute parameters
    height, width = edge_image.shape
    left_params = None
    right_params = None

    if len(points) > 0:
        # Separate points into left and right based on x-coordinates
        left_points = points[points[:, 0] < width // 2]
        right_points = points[points[:, 0] >= width // 2]

        def fit_line_params(points):
            if len(points) > 1:  # At least two points are required to fit a line
                try:
                    # Fit a polynomial to the points
                    poly_fit = np.polyfit(points[:, 1], points[:, 0], 1)  # x = m*y + c
                    slope, intercept = poly_fit
                    return slope, intercept
                except np.linalg.LinAlgError:
                    print("Not enough points to fit a line")
            return None

        # Fit left and right lines
        left_params = fit_line_params(left_points)
        right_params = fit_line_params(right_points)

    return {
        "left": left_params,  # (slope, intercept) or None
        "right": right_params  # (slope, intercept) or None
    }

def draw_lines(image, lane_lines):
    """
    Draw the left and right lane lines on the image.

    :param image: Original image to draw on.
    :param lane_lines: Dictionary with slopes and intercepts for left and right lines.
    :return: Image with lines drawn.
    """
    output_image = image.copy()
    height, width = image.shape[:2]

    for key, line in lane_lines.items():
        if line is not None:
            slope, intercept = line
            # Calculate two points on the line
            y1, y2 = 0, height  # From top to bottom of the image
            x1 = int(intercept + slope * y1)
            x2 = int(intercept + slope * y2)
            color = (0, 255, 0) if key == "left" else (255, 0, 0)  # Green for left, blue for right
            cv2.line(output_image, (x1, y1), (x2, y2), color, 2)

    return output_image

if __name__ == '__main__':
    file = 'images/rgb image_screenshot_16.12.2024.png'
    raw_frame = cv2.imread(file)
    frame = np.copy(raw_frame)
    show_image('raw image', frame)
    detect_lane(frame)
    # logging.basicConfig(level=logging.INFO)
    #
    # test_video('/home/pi/DeepPiCar/driver/data/tmp/video01')
    # test_photo('images/292.png')
    # test_photo(sys.argv[1])
    # test_video(sys.argv[1])