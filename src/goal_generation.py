#!/home/guptah1/rospythonenv/bin/python

### Python modules ###
import sys
import numpy as np

### Ros python modules ###
import tf
import rospy
import message_filters as mf
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image

PUB_RATE = 2

### To generate goal, brezier curev is used, 3pnt and 4pnt curve
def bezier_curve(p1, p2, p3):
    '''
    3 points bezier curve for new goal generation
    t = 0.25 (to be checked pratically)
    p1: curr robot point; p2:old goal; p3:currn user point
    '''
    t = 0.6
    x = pow(1-t, 2)*p1.x + 2*(1-t)*t*p2.x + pow(t,2)*p3.x
    y = pow(1-t, 2)*p1.y + 2*(1-t)*t*p2.y + pow(t,2)*p3.y
    return Point(x,y,0)

def bezier_curve_4pnts(p1, p2, p4):
        '''
        4 points bezier curve for new goal generation
        t = 0.75 (to be checked pratically)
        p1: curr robot point; p2:old goal; p4:currn user point
        '''
        t = 0.6
        
        len_PR = np.linalg.norm([p4.x - p1.x, p4.y - p1.y])
        len_OR = np.linalg.norm([p2.x - p1.x, p2.y - p1.y])
        cos_theta = ((p4.x - p1.x)*(p2.x - p1.x) + (p4.y - p1.y)*(p2.y - p1.y))/(len_PR*len_OR)
        v = len_PR*cos_theta - len_OR
        p3 = Point()
        p3.x = p4.x + v*((p1.x - p2.x)/len_OR)
        p3.y = p4.y + v*((p1.y - p2.y)/len_OR)

        x = pow(1-t, 3)*p1.x + 3*pow(1-t, 2)*t*p2.x + 3*(1-t)*pow(t,2)*p3.x + pow(t, 3)*p4.x
        y = pow(1-t, 3)*p1.y + 3*pow(1-t, 2)*t*p2.y + 3*(1-t)*pow(t,2)*p3.y + pow(t, 3)*p4.y       

        return Point(x,y,0)

class goal_generator:
    def __init__(self):
        ### Constructor
        self.marker_id = 1
        self.fixed_frame = rospy.get_param("/face_recognizer/fixed_frame", "/base_link") #"/odom_combined"
        self.r_thrs = 0.5 # threshold for distance bw old and new goal
        self.distance_bw_cob_user = 2 # # distance maintained bw robot and user
        self.theta_thrs = np.math.pi/3
        self.initialize_goal = False 
        self.should_pub_goal = False
        self.del_T = 1./(PUB_RATE)
        self.r_robot = 0.4              # robot radius (around new_goal) to see robot will fit or not
        self.free_area_threshold = 0.3  # % of area occupied at the new goal point
        self.listener = tf.TransformListener()

        ### Parameters for trajectory replication
        self.user_trajectory = []
        self.count = 0.0
        self.wait_period = 3.0 # 3 sec wait before executing trajectory replication

        ### Subscriber
        sub_person_pnt = mf.Subscriber("user_odom", Odometry)
        sub_robot_odom = mf.Subscriber("/base/odometry_controller/odometry", Odometry)
        ts = mf.ApproximateTimeSynchronizer([sub_person_pnt, sub_robot_odom], 2, slop = 0.1)
        ts.registerCallback(self.callback)
        sub_local_map = rospy.Subscriber('/move_base_node/local_costmap/costmap', OccupancyGrid, self.local_map_callback)

        ### Publisher
        self.pub_marker = rospy.Publisher("goal_markers", Marker, queue_size=1)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)


    def callback(self, person_odom, robot_odom): 
        ### Saving the user trajectory for 10 recent points
        if len(self.user_trajectory) <=10:
            self.user_trajectory.append(person_odom.pose.pose.position)
        else:
            del self.user_trajectory[0]
            self.user_trajectory.append(person_odom.pose.pose.position)

        # person position ref has to be /odom_combined
        # initializing the robot initial pose as initial goal
        if not self.initialize_goal:
            self.old_pnt = robot_odom.pose.pose.position
            _, _, self.old_theta = tf.transformations.euler_from_quaternion([robot_odom.pose.pose.orientation.x, robot_odom.pose.pose.orientation.y,
                                                                             robot_odom.pose.pose.orientation.z, robot_odom.pose.pose.orientation.w])
            self.initialize_goal = True
        
        else:
            self.now = robot_odom.header.stamp
            self.new_goal_generation(person_odom.pose.pose.position, person_odom.twist.twist.linear, robot_odom.pose.pose.position)
        
            return None
    
    def local_map_callback(self, map):
        """
        Local map callback to update our local map with a newly published one
        """
        local_map = map
        self.map_stamp = map.info.header.stamp
        self.map_frame    = map.info.header.frame_id
        self.map_pos_x = local_map.info.origin.position.x
        self.map_pos_y = local_map.info.origin.position.y
        self.res = 0.05 #local_map.info.resolution
        self.kernel_size = int(self.r_robot/self.res) ## number of cells for robot radius
        w_map = local_map.info.width
        h_map = local_map.info.height

        data = local_map.data # each cell value 0-100
        data = np.asarray(data, dtype=np.float)
        data[data==-1] = 100
        data /= 100.
        self.map_ = data.reshape((w_map, h_map)) ## data is in row-major form

    def cost_in_map(self, p): #a = (column, row)
        col = p[0]
        row = p[1]
        cost = np.sum(self.map_[row - self.kernel_size:row+self.kernel_size, col - self.kernel_size:col+self.kernel_size])/ ((2.*self.kernel_size)**2.)
        return cost # % of area occupied in the square area with side=2*robot_radius

    def get_feasible_goal_pnt(self, pnt):
        """
        Give the feaible goal for the robot to follow based on the local costmap if possible
        """
        ### Transforming pnt from odom frame to map frame 
        pnt_odom = PointStamped() # point in Camera frame
        pnt_odom.header.frame_id = self.fixed_frame
        pnt_odom.header.stamp = self.map_stamp
        pnt_odom.point = pnt
        
        pnt_map = self.listener.transformPoint(self.map_frame, pnt_odom)


        x = pnt_map.point.x
        y = pnt_map.point.y
        map_x = int(round((x - self.map_pos_x)/self.res)) ## column no.
        map_y = int(round((y - self.map_pos_y)/self.res)) ## row no.

        if self.cost_in_map([map_x, map_y]) < self.free_area_threshold: # cell corresponding to region occupied
            return pnt
        else:
            pnt_ = Point()
            shift = 0.1
            while shift<self.r_robot:
                del_d = int(shift/self.res) # if the new goal is not feasible look around
                pnt = np.array([[map_x -del_d, map_y - del_d], [map_x, map_y - del_d], [map_x +del_d, map_y - del_d],
                                [map_x - del_d, map_y], [map_x + del_d, map_y],
                                [map_x -del_d, map_y + del_d], [map_x, map_y + del_d], [map_x +del_d, map_y + del_d]])
                cost_  = np.apply_along_axis(self.cost_in_map, 1, pnt)
                if np.amin(cost_) < self.free_area_threshold:
                    pnt_.x = self.res*pnt[np.argmin(cost_)][0] + self.map_pos_x
                    pnt_.y = self.res*pnt[np.argmin(cost_)][1] + self.map_pos_y
                    
                    ### converting point from map frame to odom frame
                    pnt_map.point = pnt_
                    pnt_odom = self.listener.transformPoint(self.fixed_frame, pnt_map)
                    return pnt_odom.point
                else:
                    shift += 0.1
                    continue
        print('No feasible goal found, hence returning the original goal.')
        return pnt

    def new_goal_generation(self, person_pnt, person_vel, robot_pnt):
        new_point = Point()

        ### using predictive person pose, x = x + v*t
        person_pnt.x += self.del_T*person_vel.x
        person_pnt.y += self.del_T*person_vel.y
        
        ### current distance and angle from robot
        d = np.linalg.norm([person_pnt.x-robot_pnt.x, person_pnt.y-robot_pnt.y])
        theta = np.arctan2((person_pnt.y-robot_pnt.y), (person_pnt.x-robot_pnt.x))

        ### if user is outside radius of 2m from robot create goal on bezier curve 
        ### else create goal on line
        if (d > self.distance_bw_cob_user) and (person_pnt!=robot_pnt):
            new_point = bezier_curve(robot_pnt, self.old_pnt, person_pnt)
            new_theta = np.arctan2((person_pnt.y-new_point.y), (person_pnt.x-new_point.x))
        else:
            new_point.x = person_pnt.x - self.distance_bw_cob_user*np.cos(theta)
            new_point.y = person_pnt.y - self.distance_bw_cob_user*np.sin(theta)
            new_theta = np.arctan2((person_pnt.y-new_point.y), (person_pnt.x-new_point.x))
        
        new_point = self.old_pnt if np.linalg.norm([self.old_pnt.x - new_point.x, self.old_pnt.y - new_point.y]) < self.r_thrs else new_point
        new_theta = self.old_theta if np.abs(self.old_theta - new_theta) < self.theta_thrs else new_theta    

        ### Publishing when new_position or new_theta or both are different from old one
        if (new_point == self.old_pnt) and (new_theta == self.old_theta):
            return None
        else:
            new_point = self.get_feasible_goal_pnt(new_point) # checking the feasibility of calculated goal
            self.old_pnt = new_point
            self.old_theta  = new_theta
            self.create_goal_marker(new_point, new_theta)
            self.should_pub_goal = True
    
    def publish_goal_and_marker(self, user_status):
        ### for publishing the goal if user is tracked
        if self.should_pub_goal:
            self.pub_marker.publish(self.marker)
            self.pub_goal.publish(self.goal)
            self.should_pub_goal = False

        ### user Trajectry Replication in case user is lost and wait period is over
        elif not user_status and self.user_trajectory:
            self.count += 1.0

            if self.count/PUB_RATE > self.wait_period:
                new_point = self.user_trajectory[0]
                new_point = self.get_feasible_goal_pnt(new_point) # checking the feasibility of calculated goal
                self.old_pnt = new_point
                new_theta = self.old_theta
                self.create_goal_marker(new_point, new_theta)
                self.pub_marker.publish(self.marker)
                self.pub_goal.publish(self.goal)
                del self.user_trajectory[0]

        else:
            self.count = 0.0

    def create_goal_marker(self, pnt, theta):
        # make a visualization marker array for the occupancy grid
        # New Goal
        self.goal = PoseStamped()
        self.goal.header.stamp = self.now
        self.goal.header.frame_id = self.fixed_frame
        self.goal.pose.position = pnt
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        self.goal.pose.orientation.x = quaternion[0]
        self.goal.pose.orientation.y = quaternion[1]
        self.goal.pose.orientation.z = quaternion[2]
        self.goal.pose.orientation.w = quaternion[3]

        # Arrow Marker
        tail = pnt
        tip = Point(tail.x + 0.5*np.cos(theta), tail.y + 0.5*np.sin(theta), 0)
        self.marker = Marker()
        self.marker.action = Marker.ADD
        self.marker.header.frame_id = self.fixed_frame
        self.marker.header.stamp = self.now
        self.marker.ns = 'points_arrows'
        self.marker.id = self.marker_id
        self.marker.type = Marker.ARROW
        self.marker.scale = Vector3(0.05, 0.1, 0.2)
        self.marker.color.r = 0.5
        self.marker.color.g = 0
        self.marker.color.b = 0
        self.marker.color.a = 0.8
        self.marker.points = [ tail, tip ]


def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('goal_generator')
    gg = goal_generator()

    rate = rospy.Rate(PUB_RATE)
    while not rospy.is_shutdown():
        user_status = rospy.get_param('user_detected', False)
        gg.publish_goal_and_marker(user_status)
        rate.sleep()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)