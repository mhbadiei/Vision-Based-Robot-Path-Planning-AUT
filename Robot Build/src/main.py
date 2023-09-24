#!/usr/bin/env python

import os
import time
import math 
import rospy
import subprocess
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler
import paho.mqtt.client as mqtt
import csv
pub = None

# Define Variables
MQTT_BROKER = "192.168.43.115"
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 45
MQTT_TOPIC = "testTopic1"

# Define on_connect event Handler
def on_connect(self,mosq, obj, rc):
        #Subscribe to a the Topic
        mqttc.subscribe(MQTT_TOPIC, 0)

# Define on_subscribe event Handler
def on_subscribe(mosq, obj, mid, granted_qos):
        print("Subscribed to MQTT Topic")

# Define on_message event Handler
def on_message(mosq, obj, msg):
        global direction, acc 
        direction = msg.payload.decode().split('  ')[0]
        acc = msg.payload.decode().split('  ')[1]
        #mqttc.publish("testTopic1","VALID")


def laser_scan(data):
  regions = {
    'right':  min(min(data.ranges[30:60]), 10),
    'front':  min(min(data.ranges[0:15]+data.ranges[345:359]), 10),
    'left':   min(min(data.ranges[300:330]), 10),
  }
  set_linx_and_angz(regions)
  
def set_linx_and_angz(regions):
  mqttc.loop(.1)
  threshold_dist = 0.7
  linear_speed = 0.5
  angular_speed = 0.3

  msg = Twist()
  linear_x = 0
  angular_z = 0
  state = ''
     
  
  if regions['front'] > threshold_dist and regions['left'] > threshold_dist and regions['right'] > threshold_dist:
    state = 'No Obstacle'
    print(direction)
    if direction:
      if direction == 'right':
        linear_x = 0.2
        angular_z = -float(acc)/4
      elif direction == 'left':
        linear_x = 0.2
        angular_z = float(acc)/4      
      elif direction == 'forward':
        linear_x = float(acc)/2
        angular_z = 0          
    #linear_x = linear_speed
    #angular_z = 0
    #if regions['left']+0.15 < regions['right']:
    #    angular_z = (regions['right'] - regions['left'])/2.5 # Increase this angular speed for avoiding obstacle faster
    #elif regions['right']+0.15 < regions['left']:
    #    angular_z = (regions['right'] - regions['left'])/2.5
  elif regions['front'] < threshold_dist and regions['left'] < threshold_dist and regions['right'] < threshold_dist:
    state = 'Obstacle in Front and Left and Right'
    linear_x = -0.05#-linear_speed
    angular_z = (regions['right'] - regions['left'])/2
  elif regions['front'] < threshold_dist and regions['left'] > threshold_dist and regions['right'] > threshold_dist:
    state = 'Obstacle in Front'
    linear_x = 0
    angular_z = angular_speed
  elif regions['front'] > threshold_dist and regions['left'] > threshold_dist and regions['right'] < threshold_dist:
    state = 'Obstacle in Right'
    linear_x = 0.05
    angular_z = -angular_speed
  elif regions['front'] > threshold_dist and regions['left'] < threshold_dist and regions['right'] > threshold_dist:
    state = 'Obstacle in Left'
    linear_x = 0.05
    angular_z = angular_speed
  elif regions['front'] < threshold_dist and regions['left'] > threshold_dist and regions['right'] < threshold_dist:
    state = 'Obstacle in Front and Right'
    linear_x = 0
    angular_z = -angular_speed
  elif regions['front'] < threshold_dist and regions['left'] < threshold_dist and regions['right'] > threshold_dist:
    state = 'Obstacle in Front and Left'
    linear_x = 0
    angular_z = angular_speed
  elif regions['front'] > threshold_dist and regions['left'] < threshold_dist and regions['right'] < threshold_dist:
    state = 'Obstacle in Left and Right'
    linear_x = linear_speed
    angular_z = 0
  else:
    state = 'unknown'
    rospy.loginfo(regions)

  rospy.loginfo(state)
  msg.linear.x = linear_x
  msg.angular.z = angular_z
  pub.publish(msg)

def main():
  global pub
  
  rospy.init_node('safe_efficient_obstacle_avoidance')
  pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
  sub = rospy.Subscriber('/scan', LaserScan, laser_scan)
  rospy.spin()


if __name__ =='__main__':
    print("Please enter x position of Turtlebot for the starting position")
    x_pos = input()
    
    print("Please enter y position of Turtlebot for the starting position")
    y_pos = input()
    
    coord = [np.float32(x_pos),np.float32(y_pos)]
    print('Initial X coordinate: ', coord[0])
    print('Initial Y coordinate: ', coord[1])
    
    print("Please enter the angle for the starting position of Turtlebot in degrees")
    robot_starting_angle = [math.radians(np.float32(input()))]
    
    print('Initial starting angle Theta wrt +X axis: ', robot_starting_angle[0])   
    
    #initial_position = coord + angle
    initial_position = np.concatenate((coord,robot_starting_angle))
    
    #print('(X, Y, Theta):' ,coord[0], coord[1], angle[0])
    print('Initial pose is:-')
    print('(X, Y, Theta):', initial_position[0], initial_position[1], initial_position[2])
    
    
    q = quaternion_from_euler(0, 0, initial_position[2])
    state_msg = ModelState()
    state_msg.model_name = 'turtlebot3_burger'
    state_msg.pose.position.x = initial_position[0]
    state_msg.pose.position.y = initial_position[1]
    state_msg.pose.position.z = 0
    
    state_msg.pose.orientation.x = q[0]
    state_msg.pose.orientation.y = q[1]
    state_msg.pose.orientation.z = q[2]
    state_msg.pose.orientation.w = q[3]
    
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    resp = set_state(state_msg)
    print(resp)
    
    time.sleep(5)
    # Initiate MQTT Client
    mqttc = mqtt.Client()

    # Register Event Handlers
    mqttc.on_message = on_message
    mqttc.on_connect = on_connect
    mqttc.on_subscribe = on_subscribe
    # Connect with MQTT Broker
    mqttc.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL )

    # Continue the network loop
    #mqttc.loop_forever()
    main()
