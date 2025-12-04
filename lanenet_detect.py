import cv2
import numpy as np
from utils import warp_perspective_to_birdseye, show_video, show_image, plt_image


class LaneDetector:
    def __init__(self, model):
        self.model = model
        self.img_height = 480
        self.img_width = 640
        self.frame = None

    def detect_lane(self, frame):
        self.frame = frame

        prediction, lane_center = self.model.predict(self.frame)

        # Ensure the prediction matches the dimensions of the raw image
        if prediction.shape[:2] != self.frame.shape[:2]:
            prediction_resized = cv2.resize(prediction, (self.frame.shape[1], self.frame.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
        else:
            prediction_resized = prediction

        # visualize the predicted lines
        # show_image('', prediction_resized)

        # Transform to bird's-eye view
        birdseye_image, _ = warp_perspective_to_birdseye(prediction_resized)

        # Detect lane lines
        line_params, blue_thresh, green_thresh = self.detect_lane_lines(birdseye_image)

        # Visualize the detected lines
        birdseye_raw_image, _ = warp_perspective_to_birdseye(self.frame)
        self.visualize_lines(birdseye_raw_image, line_params)

        # # Print the line parameters
        # for color, (slope, intercept) in line_params.items():
        #     print(f"{color.capitalize()} Line: Slope={slope}, Intercept={intercept}")
        return line_params

    def detect_lane_lines(self, image):
        """
        Detect lane markings and fit lines to perceived blue, green, and red dots.
        """
        # Split the channels
        blue_channel, green_channel, red_channel = cv2.split(image)

        # Threshold each channel to isolate dots
        _, blue_thresh = cv2.threshold(blue_channel, 200, 255, cv2.THRESH_BINARY)
        _, green_thresh = cv2.threshold(green_channel, 200, 255, cv2.THRESH_BINARY)

        # Fit lines to each channel
        left_params = self.fit_line_middle(blue_thresh, side='left')
        # print('fit_line_middle(green_thresh)', fit_line_middle(green_thresh, side='right'))
        right_params = self.fit_line_middle(green_thresh, side='right')

        return {
            "left": left_params,
            "right": right_params,
        }, blue_thresh, green_thresh

    def fit_line_middle(self, binary_image, side='left'):
        """
        Filters dots close to the middle of the image and calculates the slope and intercept of the line.

        Parameters:
            binary_image: Binary image with detected dots (non-zero values).
            side: The left or right line.

        Returns:
            slope, intercept: The slope and intercept of the fitted line.
            filtered_image: Binary image with only dots close to the middle retained.
        """

        x_middle = self.img_width / 2

        # Find coordinates of non-zero pixels
        coordinates = np.column_stack(np.where(binary_image > 0))

        if len(coordinates) < 2:
            # Not enough points to fit a line
            return None, None

        # Separate x and y coordinates
        x = coordinates[:, 1]  # Horizontal positions
        y = coordinates[:, 0]  # Vertical positions

        # Filter dots based on the specified side
        if side == "left":
            side_indices = x < x_middle  # Keep dots on the left side
        elif side == "right":
            side_indices = x > x_middle  # Keep dots on the right side
        else:
            raise ValueError("Invalid side. Use 'left' or 'right'.")

        # Check if enough points remain after filtering
        if np.sum(side_indices) < 2:
            # Not enough points to fit a line
            return None, None

        # Filter the coordinates
        x_filtered = x[side_indices] # Reshape for sklearn (expects 2D array)
        y_filtered = y[side_indices]

        vx, vy, x0, y0 = cv2.fitLine(np.column_stack((x_filtered, y_filtered)), cv2.DIST_HUBER, 0, 0.01, 0.01)
        vx = vx[0]
        vy = vy[0]
        x0 = x0[0]
        y0 = y0[0]
        # print(f"vx: {vx}, vy: {vy}, x0: {x0}, y0: {y0}")

        # Normalize the direction to ensure vy >= 0
        if vy < 0:
            vx, vy = -vx, -vy

        return vx, vy, x0, y0

    def visualize_lines(self, image, line_params):
        """
        Visualize detected lines on the original image.
        """
        # Create a copy of the original image
        debug_image = image.copy()

        # Get image dimensions
        y_bottom = self.img_height - 1
        y_top = 0

        # Draw lines for each color
        for side, params in line_params.items():
            if params is None or any(p is None for p in params):
                # print(f"Skipping {side} line due to missing parameters.")
                continue  # Skip if line parameters are missing or invalid
            try:
                vx, vy, x0, y0 = params

                # Calculate slope and intercept
                slope = vy / vx if vx != 0 else None  # Avoid division by zero
                if slope is None:
                    print(f"Skipping {side} line due to vertical orientation (vx=0).")
                    continue
                intercept = y0 - slope * x0

                # Calculate x-coordinates for the line
                x_bottom = int((y_bottom - intercept) / slope)
                x_top = int((y_top - intercept) / slope)

                # Ensure coordinates are valid integers
                x_bottom = max(0, min(self.img_width - 1, x_bottom))  # Clamp to image bounds
                x_top = max(0, min(self.img_width - 1, x_top))

                # Choose line color for visualization
                line_color = (255, 0, 0) if side == "left" else (0, 255, 0) if side == "right" else (0, 0, 255)

                # Draw the line
                cv2.line(debug_image, (x_bottom, y_bottom), (x_top, y_top), line_color, 2)
            except Exception as e:
                print(f"Error while drawing {side} line: {e}")
                continue
        # show_image('debug_image', debug_image)
        show_video('debug_image', debug_image)
        # return debug_image

        # # Visualize the binary thresholded images for each color
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 4, 1)
        # plt.title("Original Image")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.subplot(1, 4, 2)
        # plt.title("Blue Threshold")
        # plt.imshow(blue_thresh, cmap="gray")
        # plt.subplot(1, 4, 3)
        # plt.title("Green Threshold")
        # plt.imshow(green_thresh, cmap="gray")
        # plt.subplot(1, 4, 4)
        # plt.title("Red Threshold")
        # plt.imshow(green_thresh, cmap="gray")
        # plt.show()
        #
        # # Show the debug image with drawn lines
        # plt.figure(figsize=(10, 5))
        # plt.title("Detected Lane Lines")
        # plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        # plt.show()

def main():
    file = 'images/292.png'
    frame = cv2.imread(file)
    from lanenet.laneNet_class import LaneNet
    model = LaneNet()
    detector = LaneDetector(model)
    detector.lane_net_detect(frame)



if __name__ == '__main__':
    file = 'rgb image_screenshot_16.12.2024.png'
    image = cv2.imread(file)
    # check_src_points(image)

    from lanenet.laneNet_class import LaneNet
    model = LaneNet()
    detector = LaneDetector(model)
    detector.lane_net_detect(image)
    # check_src_points(image)
    # detector.warp_perspective_to_birdseye(image)
    # print(detector.lane_net_detect(image))