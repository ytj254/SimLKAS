import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from modules.utils import warp_perspective_to_birdseye, show_video, show_image, plt_image
import matplotlib.pyplot as plt


class LegacyLaneDetector:
    def __init__(self):
        self.img_height = 480
        self.img_width = 640
        self.frame = None

    def detect_lane(self, frame):
        """
         Main method to detect and visualize lane markings.

        :param frame: Raw frame.
        :return: Dictionary with left and right lane line parameters.
        """
        self.frame = frame
        # plt_image(frame)

        # Step 1: Preprocess the frame
        canny_img = self.canny()

        # Step 2: Warp perspective to birdseye view
        birdseye_image, _ = warp_perspective_to_birdseye(self.frame)
        birdseye_canny, _ = warp_perspective_to_birdseye(canny_img)

        # Step 3: Detect lane marking parameters
        lane_lines = self.identify_lane_marking_params(birdseye_canny)
        # print(lane_lines)

        # Step 4: Visualize lane lines
        # visualized_image = self.draw_lines_cv2(birdseye_image, lane_lines)
        # plt_image(visualized_image)
        # show_image("Lane Marking Lines", visualized_image)

        # show_video("raw image", self.frame)
        # show_video("canny image", canny_img)
        # show_video("birdseye canny", birdseye_canny)
        # show_video("Lane Marking Lines", visualized_image)

        # Visualize the binary thresholded images for each color
        # plt.figure(figsize=(15, 3))
        # plt.subplot(1, 5, 1)
        # plt.title("Original Frame")
        # plt.imshow(self.frame)
        # plt.subplot(1, 5, 2)
        # plt.title("Gray Frame")
        # plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY), cmap="gray")
        # plt.subplot(1, 5, 3)
        # plt.title("Canny Frame")
        # plt.imshow(canny_img, cmap="gray")
        # plt.subplot(1, 5, 4)
        # plt.title("BEV Canny Frame")
        # plt.imshow(birdseye_canny, cmap="gray")
        # plt.subplot(1, 5, 5)
        # plt.title("Detected Frame")
        # plt.imshow(visualized_image)
        # plt.show()

        return lane_lines

    def canny(self):
        """
         Apply Canny edge detection to the frame.
         :return: Canny edge-detected image.
         """
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        # show_image("gray image", gray)

        # show_image("bg_subtracted image", gray)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        canny_img = cv2.Canny(blur, 60, 100)
        # show_image("canny image", canny_img)

        return canny_img

    def identify_lane_marking_params(self, edge_image, eps=15, min_samples=5):
        """
        Identify lane marking lines and compute line parameters directly using pixel points.

        :param edge_image: Binary edge-detected image.
        :param eps: DBSCAN maximum distance between points in a cluster.
        :param min_samples: Minimum number of points to form a cluster in DBSCAN.
        :return: Dictionary with left and right lane line parameters.
        """
        # Step 1: Extract nonzero pixel coordinates as points
        points = np.column_stack(np.nonzero(edge_image))  # (y, x) format

        if len(points) == 0:
            return {"left": None, "right": None}

        # Step 2: Use DBSCAN to cluster points
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        # Step 3: Separate clusters into left and right lanes
        left_clusters = []
        right_clusters = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise points
                continue

            cluster_points = points[labels == cluster_id]
            # Determine if the cluster belongs to left or right based on x-coordinates
            if np.mean(cluster_points[:, 1]) < self.img_width // 2:  # Use x-coordinates for separation
                left_clusters.append(cluster_points)
            else:
                right_clusters.append(cluster_points)

        # Step 4: Fit lines and compute parameters
        # if len(points) > 0:
        #     # Separate points into left and right based on x-coordinates
        #     left_points = points[points[:, 0] < self.img_width // 2]
        #     right_points = points[points[:, 0] >= self.img_width // 2]
        #
        #     # Fit left and right lines
        #     left_params = self.fit_line_params(left_points)
        #     right_params = self.fit_line_params(right_points)

        left_params = self.fit_line_to_clusters(left_clusters)
        right_params = self.fit_line_to_clusters(right_clusters)

        # left_params = self.fit_line_to_clusters_cv2(left_clusters)
        # right_params = self.fit_line_to_clusters_cv2(right_clusters)


        return {"left": left_params, "right": right_params}

    # Helper function to fit lines to clusters
    def fit_line_to_clusters(self, clusters):
        """
        Fit a line to the given clusters and normalize the slope direction (bottom to up).

        :param clusters: List of point clusters.
        :return: Tuple (slope, intercept) where slope is normalized to bottom-to-up direction.
        """
        all_points = np.vstack(clusters) if clusters else None
        if all_points is not None and len(all_points) > 1:

            # Fit a line using polynomial fitting
            poly_fit = np.polyfit(all_points[:, 0], all_points[:, 1], 1)  # y = mx + c
            slope, intercept = poly_fit  # Extract slope and intercept

            # return slope, intercept

            # Direction vector (vx, vy)
            vy = 1  # Always 1 since the slope is dx/dy
            vx = slope  # Slope from the polyfit output

            # A point on the line (x0, y0)
            y0 = 0  # Choose y = 0 for simplicity
            x0 = intercept  # At y = 0, x = intercept

            # Normalize the direction vector (vx, vy)
            norm = np.sqrt(vx ** 2 + vy ** 2)
            vx /= norm
            vy /= norm

            # Ensure bottom-to-top normalization (vy >= 0)
            if vy < 0:
                vx = -vx
                vy = -vy

            return vx, vy, x0, y0

        return None

    def fit_line_to_clusters_cv2(self, clusters):
        """
        Fit a line to the given clusters and normalize the slope direction (bottom to up).

        :param clusters: List of point clusters.
        :return: Tuple (slope, intercept) where slope is normalized to bottom-to-up direction.
        """
        all_points = np.vstack(clusters) if clusters else None
        if all_points is not None and len(all_points) > 1:
            # Fit a line using cv2.fitLine
            vx, vy, x0, y0 = cv2.fitLine(all_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            # Normalize the direction to ensure bottom-to-up (increasing y direction)
            if vy < 0:
                vx, vy = -vx, -vy

            # Calculate slope and intercept
            # slope = vy / vx
            # intercept = y0 - slope * x0

            return vx, vy, x0, y0

        return None

    def draw_lines(self, image, lane_lines):
        """
         Draw the left and right lane lines on the image.

         :param image: Image to draw the lines on.
         :param lane_lines: Dictionary with lane line parameters.
         :return: Image with lines drawn.
         """
        output_image = image.copy()

        for key, params in lane_lines.items():
            if params is not None:
                slope, intercept = params
                y1, y2 = 0, self.img_height  # From top to bottom of the image
                x1 = int(intercept + slope * y1)
                x2 = int(intercept + slope * y2)
                color = (0, 255, 0) if key == "left" else (255, 0, 0)  # Green for left, blue for right
                cv2.line(output_image, (x1, y1), (x2, y2), color, 5)

        return output_image

    def draw_lines_cv2(self, image, lane_lines):
        """
        Draw the left and right lane lines on the image.

        :param image: Image to draw the lines on.
        :param lane_lines: Dictionary with lane line parameters (vx, vy, x0, y0).
        :return: Image with lines drawn.
        """
        output_image = image.copy()
        # Get image dimensions
        y_bottom = self.img_height - 1
        y_top = 0

        for key, params in lane_lines.items():
            if params is not None:
                vx, vy, x0, y0 = params

                # Calculate slope and intercept
                slope = vy / vx if vx != 0 else None  # Avoid division by zero

                intercept = y0 - slope * x0

                # Calculate x-coordinates for the line
                x_bottom = int((y_bottom - intercept) / slope)
                x_top = int((y_top - intercept) / slope)

                # Ensure coordinates are valid integers
                x_bottom = max(0, min(self.img_width - 1, x_bottom))  # Clamp to image bounds
                x_top = max(0, min(self.img_width - 1, x_top))

                # Define color for left and right lines
                color = (255, 0, 0) if key == "left" else (0, 255, 0)  # Green for left, blue for right

                # Draw the line on the output image
                cv2.line(output_image, (x_top, y_top), (x_bottom, y_bottom), color, 2)

        return output_image


if __name__ == "__main__":
    file = 'images/rgb image_screenshot_16.12.2024.png'
    raw_frame = cv2.imread(file)
    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    frame = np.copy(rgb_frame)
    # show_image('raw image', frame)
    detector = LegacyLaneDetector()
    detector.detect_lane(frame)
