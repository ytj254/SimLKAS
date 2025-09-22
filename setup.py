from setuptools import find_packages, setup

package_name = 'carla_ros2_lane_keep'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ytj',
    maintainer_email='tianjiayang254@gmail.com',
    description='Lane keeping with CARLA and ROS2',
    license='Apache License 2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'subscribe_img = carla_ros2_lane_keep.subscribe_img:main',
            'lane_detection_publisher = carla_ros2_lane_keep.lane_detection_publisher:main',
        ],
    },
)
