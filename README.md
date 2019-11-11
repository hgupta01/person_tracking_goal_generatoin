# person_tracking_goal_generatoin
This package is developed during the internshipp at Aalto University for person tracking and generating goal for Care-O-bot 4
to follow.

It contains 3 Python files:
1. user_tracker_without_cart.py : this node contains a Kalman tracker for tracking a person using the measurement (distance)
from RGBD camera (person detection) and Laser Sensor (leg detection). This node is used when COB is not pushing a cart.
2. user_tracker_with_cart.py : this node contains a Kalman tracker for tracking a person using the measurement (distance)
from RGBD camera (person detection) and Laser Sensor (leg detection). This node is used when COB is pushing a cart.
3. goal_generation.py : This node generate goals for robot to follow and can be used with any robot. The node has to subscribe
to the robot odom topic and the person's position in global frame.

## How to use the package
to be updated....

