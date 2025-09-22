import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2


class CarlaImageSubscriber(Node):
    def __init__(self):
        super().__init__('carla_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',  # Update with the CARLA ROS Bridge topic
            self.image_callback,
            10  # QoS profile depth
        )
        self.subscription  # Prevent unused variable warning

    def image_callback(self, msg):
        # Convert ROS2 Image message to OpenCV format
        np_image = np.frombuffer(msg.data, dtype=np.uint8)
        bgra_image = np_image.reshape((msg.height, msg.width, 4))
        bgr_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)

        # Display the image
        cv2.imshow("CARLA Camera Image", bgr_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    carla_image_subscriber = CarlaImageSubscriber()

    try:
        rclpy.spin(carla_image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        carla_image_subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
