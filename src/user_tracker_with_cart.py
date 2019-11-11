#!/home/guptah1/rospythonenv/bin/python

### Standard Python modules
import sys
import copy
import math
import timeit
import random
import numpy as np
import scipy.stats
import scipy.spatial
from scipy.optimize import linear_sum_assignment
from pykalman import KalmanFilter # To install: http://pykalman.github.io/#installation

### ROS Python modules
import tf
import rospy
import message_filters as mf
from geometry_msgs.msg  import Point
from people_msgs.msg import People, Person
from leg_tracker.msg import Leg, LegArray 
from visualization_msgs.msg import Marker
from cob_srvs.srv import SetString
from nav_msgs.msg import Odometry

class DetectedCluster:
    """
    A detected scan cluster. Not yet associated to an existing track.
    """
    def __init__(self, pos_x, pos_y, confidence):
        """
        Constructor
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.confidence = confidence

class ObjectTracked:
    """
    A tracked object. Could be a person leg, entire person or any arbitrary object in the laser scan.
    """
    new_leg_id_num = 1

    def __init__(self, x, y, now, confidence, is_person): 
        """
        Constructor
        """        
        self.id_num = ObjectTracked.new_leg_id_num
        ObjectTracked.new_leg_id_num += 1
        self.colour = (random.random(), random.random(), random.random())
        self.last_seen = now
        self.seen_in_current_scan = True
        self.times_seen = 1
        self.confidence = confidence
        self.dist_travelled = 0.
        self.is_person = is_person

        if self.is_person:
            delta_t = 1/3. # publishing frequency of person detection
            std_process_noise = 0.08
        else:
            delta_t = 1/3.
            std_process_noise = 0.08

        std_pos = std_process_noise
        std_vel = std_process_noise
        std_obs = 0.1
        var_pos = std_pos**2
        var_vel = std_vel**2
        # The observation noise is assumed to be different when updating the Kalman filter than when doing data association
        var_obs_local = std_obs**2 
        self.var_obs = (std_obs + 0.4)**2

        # Setting the state related parameters
        self.filtered_state_means = np.array([x, y, 0, 0])
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0
        self.vel_y = 0

        self.filtered_state_covariances = 0.5*np.eye(4) # noise for the initializing the object state  

        # Constant velocity motion model
        transition_matrix = np.array([[1, 0, delta_t,        0],
                                      [0, 1,       0,  delta_t],
                                      [0, 0,       1,        0],
                                      [0, 0,       0,        1]])

        # Oberservation model. Can observe pos_x and pos_y (unless person is occluded). 
        observation_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]])

        transition_covariance = np.array([[var_pos,       0,       0,       0],
                                          [      0, var_pos,       0,       0],
                                          [      0,       0, var_vel,       0],
                                          [      0,       0,       0, var_vel]])

        observation_covariance =  var_obs_local*np.eye(2)

        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )


    def update(self, observations):
        """
        Update our tracked object with new observations
        """
        self.filtered_state_means, self.filtered_state_covariances = (
            self.kf.filter_update(
                self.filtered_state_means,
                self.filtered_state_covariances,
                observations
            )
        )

        # Keep track of the distance it's travelled 
        # We include an "if" structure to exclude small distance changes, 
        # which are likely to have been caused by changes in observation angle
        # or other similar factors, and not due to the object actually moving
        delta_dist_travelled = ((self.pos_x - self.filtered_state_means[0])**2 + (self.pos_y - self.filtered_state_means[1])**2)**(1./2.) 
        if delta_dist_travelled > 0.01: 
            self.dist_travelled += delta_dist_travelled

        self.pos_x = self.filtered_state_means[0]
        self.pos_y = self.filtered_state_means[1]
        self.vel_x = self.filtered_state_means[2]
        self.vel_y = self.filtered_state_means[3]


class UserDetection():
    def __init__(self):

        self.confidence_percentile = rospy.get_param("confidence_percentile", 0.90)
        self.mahalanobis_dist_gate = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, 1.0)
        self.fixed_frame = rospy.get_param("fixed_frame", "/base_link")
        self.user_initialised = False
        self.max_cost = 9999999
        self.confidence_threshold_to_maintain_track = rospy.get_param("confidence_threshold_to_maintain_track", 0.2)
        self.max_std = rospy.get_param("max_std", 0.95)
        self.max_cov = self.max_std**2
        self.cob_say =  rospy.ServiceProxy('/sound/say', SetString)
        self.leg_tracked = []
        self.detected_leg_clusters = []
        self.use_person_measure = False

        self.robot_x = 0.01
        self.robot_y = 0.0

        sub_face = rospy.Subscriber("face_detect", People, self.face_msg_callback)
        sub_person = rospy.Subscriber("people_tf_detect", People, self.person_msg_callback)
        sub_robot_odom = rospy.Subscriber("/base/odometry_controller/odometry", Odometry, self.robot_odom_callback)
        sub_detected_clusters = rospy.Subscriber('detected_leg_clusters', LegArray, self.leg_clusters_callback) 

        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=5)
        self.odom_pub = rospy.Publisher('user_odom', Odometry, queue_size=1)
    
    def robot_odom_callback(self, odom):
        self.robot_x = odom.pose.pose.position.x
        self.robot_y = odom.pose.pose.position.y
         
    def face_msg_callback(self, face_pnts):
        if len(face_pnts.people)>1:
            conf = np.empty(len(face_pnts.people), np.float)
            for i, face in enumerate(face_pnts.people):
                conf[i] = face.reliability
            user_face = np.array([face_pnts.people[np.argmax(conf)].position.x, face_pnts.people[np.argmax(conf)].position.y])
        else:
            user_face = np.array([face_pnts.people[0].position.x, face_pnts.people[0].position.y])
        
        if not self.user_initialised:
            self.cob_say('User Detected')
            rospy.set_param('user_detected', True)
            print("user initisalized: ", user_face[0], user_face[1])
            self.user_track = ObjectTracked(user_face[0], user_face[1], face_pnts.header.stamp, 1., is_person=True) 
            self.user_initialised = True
            
        ## if face is available and user is already tracked
        ## check if face and tracking data are within threshold to each other 
        else:
            dist = np.linalg.norm([user_face[0]-self.user_track.pos_x, user_face[1]-self.user_track.pos_y])
            if dist > 0.4:
                self.user_track = ObjectTracked(user_face[0], user_face[1], face_pnts.header.stamp, 1., is_person=True)
            
    def match_detections_to_tracks_GNN(self, objects_tracked, objects_detected):
        """
        Match detected objects to existing object tracks using a global nearest neighbour data association
        """
        matched_tracks = {}

        # Populate match_dist matrix of mahalanobis_dist between every detection and every track
        match_dist = [] # matrix of probability of matching between all people and all detections.   
        eligable_detections = [] # Only include detections in match_dist matrix if they're in range of at least one track to speed up munkres
        for detect in objects_detected: 
            at_least_one_track_in_range = False
            new_row = []
            for track in objects_tracked:
                # Use mahalanobis dist to do matching
                cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                mahalanobis_dist = math.sqrt(((detect.pos_x-track.pos_x)**2 + (detect.pos_y-track.pos_y)**2)/cov) # = scipy.spatial.distance.mahalanobis(u,v,inv_cov)**2
                if mahalanobis_dist < self.mahalanobis_dist_gate:
                    cost = mahalanobis_dist
                    at_least_one_track_in_range = True
                else:
                    cost = self.max_cost 
                new_row.append(cost)                    
            # If the detection is within range of at least one track, add it as an eligable detection in the munkres matching 
            if at_least_one_track_in_range: 
                match_dist.append(new_row)
                eligable_detections.append(detect)

        # Run munkres on match_dist to get the lowest cost assignment
        if match_dist:
            elig_detect_indexes, track_indexes = linear_sum_assignment(match_dist)
            for elig_detect_idx, track_idx in zip(elig_detect_indexes, track_indexes):
                if match_dist[elig_detect_idx][track_idx] < self.mahalanobis_dist_gate:
                    detect = eligable_detections[elig_detect_idx]
                    track = objects_tracked[track_idx]
                    matched_tracks[track] = detect
        return matched_tracks

    def person_msg_callback(self, person_pnts):
        if self.user_initialised:
            # making list of person detected
            person_detected = []
            for person in person_pnts.people:
                person_detected.append(DetectedCluster(person.position.x, person.position.y, person.reliability))

            user_propogated = copy.deepcopy(self.user_track)
            user_propogated.update(np.ma.masked_array(np.array([0, 0]), mask=[1,1]))
                         
            cost = np.empty(len(person_detected), np.float)
            at_least_one_track_in_range = False
            for i, detect in enumerate(person_detected):
                cov = user_propogated.filtered_state_covariances[0][0] + user_propogated.var_obs # cov_xx == cov_yy == cov
                mahalanobis_dist = math.sqrt(((detect.pos_x-user_propogated.pos_x)**2 + (detect.pos_y-user_propogated.pos_y)**2)/cov) # = scipy.spatial.distance.mahalanobis(u,v,inv_cov)**2
                if mahalanobis_dist < self.mahalanobis_dist_gate:
                    at_least_one_track_in_range = True
                    cost[i] = mahalanobis_dist
                else:
                    cost[i] = self.max_cost

            if at_least_one_track_in_range:
                self.person_measure = person_detected[np.argmin(cost)]
                theta = abs(np.math.atan2((self.person_measure.pos_y - self.robot_y), (self.person_measure.pos_x - self.robot_x)) - 
                            np.math.atan2(self.robot_y, self.robot_x))
                if theta < (25*np.pi/180):
                    self.use_person_measure = True
                else:
                    self.use_person_measure = False
            else:
                self.use_person_measure = False
    
    def leg_clusters_callback(self, detected_clusters_msg):
        self.now = detected_clusters_msg.header.stamp
        self.detected_leg_clusters = []
        ## confidence based on pretrained Random Forest tree model
        for cluster in detected_clusters_msg.legs:
            if cluster.confidence > 0.05:
                new_detected_cluster = DetectedCluster(
                    cluster.position.x, 
                    cluster.position.y, 
                    cluster.confidence, 
                )      
                self.detected_leg_clusters.append(new_detected_cluster)
    
    def update_leg_tracks(self, object_tracked, leg_detect):
        # Propogate existing tracks
        to_duplicate = set()
        propogated = copy.deepcopy(object_tracked)
        for propogated_track in propogated:
            propogated_track.update(np.ma.masked_array(np.array([0, 0]), mask=[1,1])) 
            if propogated_track.is_person:
                to_duplicate.add(propogated_track)
    
        # Duplicate tracks of user so it can be matched twice in the matching
        duplicates = {}
        for propogated_track in to_duplicate:
            propogated.append(copy.deepcopy(propogated_track))
            duplicates[propogated_track] = propogated[-1]

        # Match detected objects to existing tracks
        matched_tracks = self.match_detections_to_tracks_GNN(propogated, leg_detect) # size of propogated is 2 with both are user_track

        tracks_to_delete = set()   
        for idx, track in enumerate(object_tracked): 
            propogated_track = propogated[idx] # Get the corresponding propogated track, not iterating through duplicate person
            if propogated_track.is_person:
                if propogated_track in matched_tracks and duplicates[propogated_track] in matched_tracks:
                    # Two matched legs for this person. Create a new detected cluster which is the average of the two
                    md_1 = matched_tracks[propogated_track]
                    md_2 = matched_tracks[duplicates[propogated_track]]
                    matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., (md_1.confidence+md_2.confidence)/2.)
                elif propogated_track in matched_tracks:
                    # Only one matched leg for this person
                    md_1 = matched_tracks[propogated_track]
                    md_2 = duplicates[propogated_track]
                    matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., md_1.confidence)                    
                elif duplicates[propogated_track] in matched_tracks:
                    # Only one matched leg for this person 
                    md_1 = matched_tracks[duplicates[propogated_track]]
                    md_2 = propogated_track
                    matched_detection = DetectedCluster((md_1.pos_x+md_2.pos_x)/2., (md_1.pos_y+md_2.pos_y)/2., md_1.confidence)                                        
                else:      
                    # No legs matched for this person 
                    matched_detection = None  
            else:
                if propogated_track in matched_tracks:
                    # Found a match for this non-person track
                    matched_detection = matched_tracks[propogated_track]
                else:
                    matched_detection = None  

            if matched_detection:
                observations = np.array([matched_detection.pos_x, 
                                         matched_detection.pos_y])
                track.confidence = 0.95*track.confidence + 0.05*matched_detection.confidence                                       
                track.times_seen += 1
                track.last_seen = self.now
                track.seen_in_current_scan = True
            else: # propogated_track not matched to a detection
                observations = np.ma.masked_array(np.array([0, 0]), mask=[1,1]) 
                track.seen_in_current_scan = False
                        
            # Input observations to Kalman filter
            track.update(observations)

            if not track.is_person:
                # Check track for deletion because covariance is too large
                cov = track.filtered_state_covariances[0][0] + track.var_obs # cov_xx == cov_yy == cov
                if cov > self.max_cov:
                    tracks_to_delete.add(track)

        for track in tracks_to_delete:
            object_tracked.remove(track)
            
        # If detections were not matched, create a new track  
        for detect in leg_detect:      
            if not detect in matched_tracks.values():
                object_tracked.append(ObjectTracked(detect.pos_x, detect.pos_y, self.now, detect.confidence, is_person=False))

        return object_tracked

    def update_and_publish_user_track(self):
        if self.user_initialised:
            if self.use_person_measure:
                print('inside person_tracker')
                self.user_track.update(np.array([self.person_measure.pos_x, self.person_measure.pos_y]))
                self.user_track.confidence = 0.95*self.user_track.confidence + 0.05*self.person_measure.confidence                                       
                self.user_track.times_seen += 1
                self.user_track.last_seen = self.now
                self.user_track.seen_in_current_scan = True
                self.leg_tracked = self.update_leg_tracks(copy.deepcopy(self.leg_tracked), copy.deepcopy(self.detected_leg_clusters))
                self.use_person_measure = False

            else:
                print('inside leg_tracker')
                object_tracked = [copy.deepcopy(self.user_track)]
                object_tracked.extend(copy.deepcopy(self.leg_tracked))
                object_tracked = self.update_leg_tracks(object_tracked, self.detected_leg_clusters)
                self.user_track = object_tracked[0]
                self.leg_tracked = object_tracked[1:]

            self.publish_tracked_people()

            # Check track for deletion           
            if  self.user_track.confidence < self.confidence_threshold_to_maintain_track:
                self.user_initialised = False
                self.cob_say('User lost')
                rospy.loginfo("deleting due to low confidence")
            else:
                # Check track for deletion because covariance is too large
                cov = self.user_track.filtered_state_covariances[0][0] + self.user_track.var_obs # cov_xx == cov_yy == cov
                if cov > self.max_cov:
                    self.cob_say('User lost')
                    self.user_initialised = False
                    rospy.loginfo("deleting due to low cov")
                    
        else:
            self.leg_tracked = self.update_leg_tracks(copy.deepcopy(self.leg_tracked), copy.deepcopy(self.detected_leg_clusters))
   
    def publish_tracked_people(self):
        """
        Publish markers of tracked people to Rviz and to <people_tracked> topic
        """        
        person_odom = Odometry()
        person_odom.header.stamp = self.now
        person_odom.header.frame_id = self.fixed_frame
        person_odom.pose.pose.position.x = self.user_track.pos_x
        person_odom.pose.pose.position.y = self.user_track.pos_y
        person_odom.twist.twist.linear.x = self.user_track.vel_x
        person_odom.twist.twist.linear.y = self.user_track.vel_y
        self.odom_pub.publish(person_odom)

        marker_id = 1
        # publish rviz markers       
        # Cylinder for body 
        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = self.now
        marker.ns = "People_tracked"
        marker.color.r = self.user_track.colour[0]
        marker.color.g = self.user_track.colour[1]
        marker.color.b = self.user_track.colour[2]          
        marker.color.a = (rospy.Duration(3) - (rospy.get_rostime() - self.user_track.last_seen)).to_sec()/rospy.Duration(3).to_sec() + 0.1
        marker.pose.position.x = self.user_track.pos_x
        marker.pose.position.y = self.user_track.pos_y
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 1.2
        marker.pose.position.z = 0.8
        self.marker_pub.publish(marker)

        # Sphere for head shape                        
        marker.type = Marker.SPHERE
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2                
        marker.pose.position.z = 1.5
        marker.id = marker_id 
        marker_id += 1                        
        self.marker_pub.publish(marker)  

        # Text showing person's ID number 
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = str(self.user_track.id_num)
        marker.scale.z = 0.2         
        marker.pose.position.z = 1.7
        self.marker_pub.publish(marker)

        # Arrow pointing in direction they're facing with magnitude proportional to speed
        marker.color.r = self.user_track.colour[0]
        marker.color.g = self.user_track.colour[1]
        marker.color.b = self.user_track.colour[2]          
        marker.color.a = (rospy.Duration(3) - (rospy.get_rostime() - self.user_track.last_seen)).to_sec()/rospy.Duration(3).to_sec() + 0.1                        
        start_point = Point()
        end_point = Point()
        start_point.x = marker.pose.position.x 
        start_point.y = marker.pose.position.y 
        end_point.x = start_point.x + 0.5*self.user_track.vel_x
        end_point.y = start_point.y + 0.5*self.user_track.vel_y
        marker.pose.position.x = 0.
        marker.pose.position.y = 0.
        marker.pose.position.z = 0.1
        marker.id = marker_id
        marker_id += 1
        marker.type = Marker.ARROW
        marker.points.append(start_point)
        marker.points.append(end_point)
        marker.scale.x = 0.05
        marker.scale.y = 0.1
        marker.scale.z = 0.2
        self.marker_pub.publish(marker) 

        # <self.confidence_percentile>% confidence bounds of person's position as an ellipse:
        cov = self.user_track.filtered_state_covariances[0][0] + self.user_track.var_obs # cov_xx == cov_yy == cov
        std = cov**(1./2.)
        gate_dist_euclid = scipy.stats.norm.ppf(1.0 - (1.0-self.confidence_percentile)/2., 0, std)
        marker.pose.position.x = self.user_track.pos_x
        marker.pose.position.y = self.user_track.pos_y            
        marker.type = Marker.SPHERE
        marker.scale.x = 2*gate_dist_euclid
        marker.scale.y = 2*gate_dist_euclid
        marker.scale.z = 0.01   
        marker.color.r = self.user_track.colour[0]
        marker.color.g = self.user_track.colour[1]
        marker.color.b = self.user_track.colour[2]            
        marker.color.a = 0.1
        marker.pose.position.z = 0.0
        marker.id = marker_id 
        marker_id += 1                    
        self.marker_pub.publish(marker)        

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('tracking')
    ud = UserDetection()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        ud.update_and_publish_user_track()
        rate.sleep()

    rospy.spin()

        
if __name__ == '__main__':
    main(sys.argv)
