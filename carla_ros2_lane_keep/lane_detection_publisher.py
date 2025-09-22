import sys
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
import cv2

# Dynamically add the external path
external_model_path = "/home/ytj/carla-ros-bridge/src/carla_ros2_lane_keep/carla_ros2_lane_keep/models"  # Replace with the actual path
if external_model_path not in sys.path:
    sys.path.append(external_model_path)

class LaneDetection(Node):
    def __init__(self, model):
        super().__init__('lane_detection_node')
        self.model = model

        # Subscriber to the raw camera image
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',  # Topic to subscribe to
            self.callback,
            10  # QoS profile depth
        )

        # Publisher for the processed image
        self.publisher = self.create_publisher(
            Image,
            '/lka/detected_image',  # Topic to publish to
            10  # QoS profile depth
        )

    def callback(self, raw_image):
        try:
            # Convert ROS2 Image message to a NumPy array
            byte_image = raw_image.data
            np_image = np.frombuffer(byte_image, dtype=np.uint8)
            bgra_image = np_image.reshape((raw_image.height, raw_image.width, 4))
            rgb_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)

            # Use the model to predict the lane and lane center
            prediction, lane_center = self.model.predict(rgb_image)

            # Ensure the prediction matches the dimensions of the raw image
            if prediction.shape[:2] != rgb_image.shape[:2]:
                prediction_resized = cv2.resize(prediction, (rgb_image.shape[1], rgb_image.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
            else:
                prediction_resized = prediction

            # Blend the overlay with the original image
            overlay_image = cv2.addWeighted(rgb_image, 0.6, prediction_resized, 5.0, 0)

            # Convert the blended image to BGRA format for ROS2 Image message
            overlay_bgra = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGRA)

            # Prepare the processed image as a ROS2 Image message
            publish_image = Image()
            publish_image.header = raw_image.header
            publish_image.height = overlay_bgra.shape[0]
            publish_image.width = overlay_bgra.shape[1]
            publish_image.encoding = raw_image.encoding  # Adjust encoding as needed
            publish_image.is_bigendian = raw_image.is_bigendian
            publish_image.step = overlay_bgra.shape[1] * 4  # width * byte_depth * num_channels
            publish_image.data = overlay_bgra.tobytes()

            # Publish the processed image
            self.publisher.publish(publish_image)
            # self.get_logger().info("Published processed image")

            # # Optionally display the overlay image locally
            # cv2.imshow("Overlay Image", overlay_image)
            # cv2.waitKey(1)  # Refresh the OpenCV window

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()  # Initialize ROS2

    from laneNet_class import LaneNet
    model = LaneNet()  # Load your lane detection model

    # Create the ROS2 node
    ros_node = LaneDetection(model)

    # Spin the node
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
