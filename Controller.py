"""
This module is developed based on the PID controller in carla documentation.
It contains PID controller to perform longitudinal control and PID or Stanley to perform lateral control
"""

from collections import deque
import math
import numpy as np
import carla


class VehicleController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(self, vehicle, method='pid', max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = LongitudinalController(self._vehicle)
        self._lat_controller = LateralController(self._vehicle, method)
        print(f'-----{method} is used.------')

    def run_step(self, target_speed, lane_params):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to keep lane center at a given target_speed.

            :param target_speed: desired vehicle speed
            :param lane_params: detected lane parameters include slope and intercept.
        """
        current_speed = self.get_speed()
        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        current_steering = self._lat_controller.run_step(current_speed, target_speed, lane_params)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1
        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = float(steering)
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        # print('current steering:', steering)

        return control

    def get_speed(self):
        """
        Compute speed of a vehicle in Km/h.

            :return: speed as a float in Km/h
        """
        vel = self._vehicle.get_velocity()

        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


class LongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self.current_speed = None
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param current_speed: current speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class LateralController:
    """
    PIDLateralController implements lateral control using a PID.
    80 km/h K_P=0.1, K_I=0.1, K_D=0.01;
    60 km/h K_P=0.1, K_I=0.1, K_D=0.01;
    40 km/h K_P=0.1, K_I=0.1, K_D=0.01;
    30 km/h K_P=0.2, K_I=0.1, K_D=0.01; off, high beam
    """

    def __init__(self, vehicle, method='pid', dynamic=None, K_P=0.1, K_I=0.1, K_D=0.01, dt=0.05, lane_width_px=420, lane_width_m=3.5):
        """
        :param vehicle: Vehicle actor to control.
        :param method: Control method includes pid, stanley.
        :param K_P: Proportional gain.
        :param K_I: Integral gain.
        :param K_D: Derivative gain.
        :param dt: Time step for the PID controller.
        :param lane_width_px: Lane width in pixels (e.g., 420 pixels).
        :param lane_width_m: Real-world lane width in meters (e.g., 3.5 meters).
        """
        self._vehicle = vehicle
        self._method = method
        self._dynamic = dynamic
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._lane_center_pixels = 320
        self._lane_width_px = lane_width_px
        self._lane_width_m = lane_width_m
        self._pixel_to_meter = lane_width_m / lane_width_px  # Scaling factor
        self._e_buffer = deque(maxlen=10)  # Error buffer for PID
        self._left_lane_history = deque(maxlen=2)  # History of left lane slopes
        self._right_lane_history = deque(maxlen=2)  # History of right lane slopes
        self._prev_left_lane = None
        self._prev_right_lane = None

    def run_step(self, current_speed, target_speed, lane_params):
        """
        Calculate steering control based on lane parameters.

        :param current_speed: current speed in Km/h
        :param target_speed: target speed in Km/h
        :param lane_params: Dictionary with two tuples (slope, intercept) for the left and right lane, or None if not detected.
        :return: Steering control value [-1, 1].
        """
        # Step 1: Calculate the centerline position in pixels
        left_lane = lane_params['left']
        right_lane = lane_params['right']

        # Step 2: Constrain the slopes of the lanes
        constrained_left_lane = self._constrain_lane(left_lane, self._prev_left_lane)
        constrained_right_lane = self._constrain_lane(right_lane, self._prev_right_lane)

        self._prev_left_lane = constrained_left_lane
        self._prev_right_lane = constrained_right_lane

        y_lookahead = -5 * target_speed + 700  # Lookahead distance, adaptive with the speed (y-coordinate in pixels).
        center_x_pixels = self._calculate_centerline(constrained_left_lane, constrained_right_lane, y_lookahead)
        if center_x_pixels is None:
            print("No lanes detected, maintaining previous steering.")
            return 0.0  # Default to straight steering when no lanes are detected

        # Step 2: Calculate the center offset in meters
        center_offset_pixels = center_x_pixels - self._lane_center_pixels
        center_offset_meters = center_offset_pixels * self._pixel_to_meter  # Convert to meters

        # Step 3: Compute angle deviation
        angle_deviation = self._calculate_angle_deviation(constrained_left_lane, constrained_right_lane)

        # Step 4: Combine center offset and angle deviation

        combined_error = center_offset_meters * 0.5 + angle_deviation
        # print(f'CV2 center_offset_meters: {center_offset_meters}, angle_deviation: {angle_deviation}')

        # Step 5: Apply control
        if self._method == 'pid':
            steering = self._pid_control(combined_error, current_speed, target_speed)
        else:
            steering = self._stanley_control(center_offset_meters, angle_deviation, current_speed)

        return np.clip(steering, -1.0, 1.0)  # Clamp the steering output

    def _constrain_lane(self, current_lane, prev_lane):
        """
        Constrain lane slope changes to avoid large deviations.

        :param current_lane: Current lane parameters (vx, vy, x0, y0) or None.
        :param lane_history: Deque storing the history of previous lane parameters.
        :return: Constrained lane parameters.
        """
        # print(f'previous: {prev_lane}, current: {current_lane}')
        # Check if current_lane is invalid
        if current_lane is None or any(val is None for val in current_lane):
            return None

        # print(current_lane)
        vx, vy, x0, y0 = current_lane
        current_slope = math.atan2(vy, vx)

        if prev_lane is not None and all(val is not None for val in prev_lane):
            # Get the previous slope
            prev_slope = math.atan2(prev_lane[1], prev_lane[0])
            slope_change = abs(current_slope - prev_slope)

            if slope_change > math.radians(30):  # 30 degrees in radians
                print(f"Excessive slope change detected: {math.degrees(slope_change):.2f} degrees.")
                # return prev_lane
                avg_slope = (current_slope + prev_slope) / 2
                vx = math.cos(avg_slope)
                vy = math.sin(avg_slope)

        # return current_lane
        return vx, vy, x0, y0

    def _calculate_centerline(self, left_lane, right_lane, y_lookahead):
        """
        Calculate the target centerline x-coordinate in pixels.

        :param left_lane: Tuple (vx, vy, x0, y0) or None.
        :param right_lane: Tuple (vx, vy, x0, y0) or None.
        :param y_lookahead: y-coordinate for lookahead in pixels.
        :return: x-coordinate of the centerline.
        """
        left_x = None
        right_x = None

        # Calculate x for the left lane
        if left_lane and all(val is not None for val in left_lane):
            vx, vy, x0, y0 = left_lane
            if vy != 0:  # Avoid division by zero
                left_x = x0 + (vx / vy) * (y_lookahead - y0)

        # Calculate x for the right lane
        if right_lane and all(val is not None for val in right_lane):
            vx, vy, x0, y0 = right_lane
            if vy != 0:  # Avoid division by zero
                right_x = x0 + (vx / vy) * (y_lookahead - y0)

        # Case 1: Both lanes detected
        if left_x is not None and right_x is not None:
            return (left_x + right_x) / 2  # Midpoint of the lanes

        # Case 2: Only left lane detected
        elif left_x is not None:
            return left_x + self._lane_width_px / 2  # Estimate centerline to the right

        # Case 3: Only right lane detected
        elif right_x is not None:
            return right_x - self._lane_width_px / 2  # Estimate centerline to the left

        # Case 4: No lanes detected
        return None

    def _calculate_angle_deviation(self, left_lane, right_lane):
        """
        Calculate the angle deviation between the vehicle heading and the lane direction.

        :param left_lane: Tuple (vx, vy, x0, y0) or None.
        :param right_lane: Tuple (vx, vy, x0, y0) or None.
        :return: Angle deviation in radians.
        """
        lane_angle = None
        # print(f'left: {left_lane}, right: {right_lane}')

        # Calculate average lane direction if both lanes are detected
        if left_lane and all(val is not None for val in left_lane) and right_lane and all(
                val is not None for val in right_lane):
            left_angle = math.atan2(left_lane[1], left_lane[0])  # atan2(vy, vx)
            right_angle = math.atan2(right_lane[1], right_lane[0])  # atan2(vy, vx)
            lane_angle = (left_angle + right_angle) / 2

        # Use left lane direction if right lane is missing
        elif left_lane and all(val is not None for val in left_lane):
            lane_angle = math.atan2(left_lane[1], left_lane[0])  # atan2(vy, vx)

        # Use right lane direction if left lane is missing
        elif right_lane and all(val is not None for val in right_lane):
            lane_angle = math.atan2(right_lane[1], right_lane[0])  # atan2(vy, vx)

        # If no lanes are detected, return zero deviation
        if lane_angle is not None:
            # Vehicle heading in bird's-eye view is fixed at π/2 radians (vertical line)
            vehicle_heading = math.pi / 2

            # Angle deviation
            angle_deviation = lane_angle - vehicle_heading
            # print(f'lane angle: {lane_angle}, vehicle heading: {vehicle_heading}')

            # Normalize the angle deviation to the range [-pi, pi]
            angle_deviation = (angle_deviation + math.pi) % (2 * math.pi) - math.pi
            return angle_deviation

        return 0.0  # No valid lanes detected

    def _dynamic_pid_parameters(self, current_speed, target_speed):
        """
        Adjust PID parameters dynamically based on speed.

        :param current_speed: Current speed of the vehicle km/h.
        :param target_speed: Target speed km/h.
        :return: Tuple of (Kp, Ki, Kd) for the current speed.
        """
        # Normalize speed (0 to 1 range)
        normalized_speed = min(current_speed / target_speed, 1.0)

        # Scale PID parameters
        Kp = self._k_p * (1 - normalized_speed)  # Decrease Kp as speed increases
        Ki = self._k_i * normalized_speed  # Slightly increase Ki with speed
        Kd = self._k_d * (1 - normalized_speed)  # Decrease Kd as speed increases

        return Kp, Ki, Kd

    def _pid_control(self, error, current_speed, target_speed):
        """
        Apply PID control logic.

        :param error: Combined lateral and angle error.
        :return: Steering control value.
        """
        self._e_buffer.append(error)

        if len(self._e_buffer) >= 2:
            de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            ie = sum(self._e_buffer) * self._dt
        else:
            de = 0.0
            ie = 0.0
        if self._dynamic:
            Kp, Ki, Kd = self._dynamic_pid_parameters(current_speed, target_speed)
            return (Kp * error) + (Ki * ie) + (Kd * de)
        else:
            return (self._k_p * error) + (self._k_i * ie) +(self._k_d * de)

    def _stanley_control(self, lateral_error, heading_deviation, current_speed):
        k = 0.01
        epsilon = 1e-6
        second_term = math.atan2(k * lateral_error, current_speed+epsilon)
        print(f'first term: {heading_deviation}, second term: {second_term}')
        return heading_deviation + math.atan2(k * lateral_error, current_speed+epsilon)
