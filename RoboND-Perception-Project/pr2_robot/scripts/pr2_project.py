#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

TEST_SCENE_VARIABLE = 3
OUTPUT_FILENAME = "output_" + str(TEST_SCENE_VARIABLE) + '.yaml'

def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def outlier_removal_filter(pcl_cloud_noise, mean = 20, std_dev = 0.5):
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = pcl_cloud_noise.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(mean)

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(std_dev)

    # Finally call the filter function for magic
    outlier_removal= outlier_filter.filter()

    return outlier_removal

def color_assignment(cluster_indices, white_cloud):

    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    return color_cluster_point_list

def get_cluster_indices(white_cloud, tolerance = 0.05, minsize=100, maxsize = 1500):

    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    #    NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(minsize)
    ec.set_MaxClusterSize(maxsize)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    return cluster_indices

def cloud_passthrough_filter(pcl_filter, filter_axis = 'z', axis_min = 0.6, axis_max = 1.1 ):

    passthrough = pcl_filter.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    return cloud_filtered

def cloud_segmentation(pcl_cloud):

    # TODO: Voxel Grid Downsampling
    vox = pcl_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005 
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # TODO: PassThrough Filter
    pcl_filtered_z = cloud_passthrough_filter(pcl_filter = cloud_filtered ,filter_axis = 'z', axis_min = 0.6, axis_max = 1.3 )
    pcl_filtered_xz = cloud_passthrough_filter(pcl_filter = pcl_filtered_z ,filter_axis = 'x', axis_min = 0.3, axis_max = 1.0 )
    pcl_filtered = cloud_passthrough_filter(pcl_filter = pcl_filtered_xz ,filter_axis = 'y', axis_min = -0.5, axis_max = 0.5 )
    
    # TODO: RANSAC Plane Segmentation
    seg = pcl_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliers, coefficients = seg.segment()
    cloud_objects = pcl_filtered.extract(inliers, negative=True)
    cloud_table = pcl_filtered.extract(inliers, negative=False)

    return cloud_objects, cloud_table

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud_noise = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    pcl_cloud = outlier_removal_filter(pcl_cloud_noise, mean = 50, std_dev = 0.5)

    # TODO: Segmentation 
    cloud_objects, cloud_table = cloud_segmentation(pcl_cloud)
    
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)# Apply function to convert XYZRGB to XYZ
    cluster_indices = get_cluster_indices(white_cloud, tolerance = 0.01, minsize=200, maxsize = 15000)
	
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    color_cluster_point_list = color_assignment(cluster_indices, white_cloud)
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
	pcl_ros = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(pcl_ros, using_hsv=True)
        normals = get_normals(pcl_ros)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .3
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

# Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    test_scene_num.data = TEST_SCENE_VARIABLE 
    dict_list = []
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')
    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the object_list
    for objects in object_list:
        object_name.data = str(objects.label)
        points_arr = ros_to_pcl(objects.cloud).to_array()
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        x, y, z = np.mean(points_arr, axis = 0)[:3]
        print "Object name : ", object_name.data
        print "In position : "
        pick_pose.position.x = np.asscalar(x) 
        pick_pose.position.y = np.asscalar(y)
        pick_pose.position.z = np.asscalar(z)
        print pick_pose.position 

        for pick_list in object_list_param :
            if pick_list['name'] == object_name.data :
	       group = pick_list['group']
               
        # TODO: Create 'place_pose' for the object
               for dropbox in dropbox_list_param:
                   if dropbox['group'] == group :
                      print "Group", group
                      a, b, c = dropbox['position']
      	              place_pose.position.x = np.float(a) 
                      place_pose.position.y = np.float(b)
                      place_pose.position.z = np.float(c)
        # TODO: Assign the arm to be used for pick_place      
                      arm_name.data = dropbox['name']
                      print "Goes to"
                      print place_pose.position
                      print "Arm", arm_name.data
                      break
               break
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        #try:
         #   pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
          #  resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

           # print ("Response: ",resp.success)

        #except rospy.ServiceException, e:
         #   print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(OUTPUT_FILENAME, dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_recognition', anonymous = True)

    # TODO: Create Subscribers
    sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)


    # TODO: Load Model From disk
    model = pickle.load(open('/home/andy/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
	rospy.spin()





