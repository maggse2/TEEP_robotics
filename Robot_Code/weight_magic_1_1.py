#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils import getWalkingParams, Robot
from vision import *
from copy import copy
import sys
from std_msgs.msg import Int32, String, Float32, Float64, Bool, Float32MultiArray
from sensor_msgs.msg import JointState
#import matplotlib.pyplot as plt
import time
#from matplotlib.animation import FuncAnimation
#from scipy import stats
from pygame import mixer
from itertools import count
#import pandas as pd
import os
#import mediapipe as mp
#import finger_recognition.FingerCounterModule as fcm

DEBUG_MODE = False # Show the detected image

MIN_AREA = 100 # Minimum area of objects to consider for vision
BALL_PICKUP_SIZE = 9000
BASKET_DUNK_SIZE = 58000
SIDE_STEP_TIME = 3.5
HEAD_SEARCH_SPEED = 0.065

class States:

    INIT = -1
    READY = 0 # Waits for start button
    RAISE_RIGHT_HAND = 50
    READ = 100
    TEST = 110
    FIND_BAG = 1
    FOCUS_BAG = 2
    PICK_BAG = 3
    FACE_BAG = 4
    TALK_USR = 5
    INPUT_USR =6
    WAIT_BAG = 7 #Waiting for bag delivery
    REMOVE_BAG = 8 #Waiting for bag takeoff


# Functions to be passed to vision system
func1 = detectSingleColor
func2 = detect2Color
args1 = ((np.array([168, 60, 0]), np.array([180, 255, 255])),)# hsv value for ping pong
#0,130,0
args2 = ((np.array([0, 120, 0]), np.array([12, 255, 255])),
         (np.array([168, 60, 0]), np.array([180, 255, 255])))# hsv value for basket

# # Create vision system
vision = VisionSystem(pipeline_funcs=[func1, func2],
                      pipeline_args=[args1, args2], debug=DEBUG_MODE, verbose=0)

# # Subscribe to cv_camera topic with vision system
rospy.Subscriber("/cv_camera/image_raw", Image, vision.read, queue_size=1)
l_grip_pub = rospy.Publisher('/grippers/left_pos', Float32, queue_size = 1)
r_grip_pub = rospy.Publisher('/grippers/right_pos', Float32, queue_size = 1)

# Create robot
robot = Robot()


# Iinitialize Node
rospy.init_node("weight_magic_trick")

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages

def set_gripper(gripper = 'left', value = 0):
    if gripper == 'left':
        for i in range(4):
            l_grip_pub.publish(value)
            rospy.sleep(0.1)
    elif gripper == 'right':
        for i in range(4):
            r_grip_pub.publish(value)
            rospy.sleep(0.1)
    else:
        rospy.info('wrong name assigned to grippers')
    rospy.sleep(0.5)


def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position
    robot.setGeneralControlModule("action_module")

    '''robot.setGrippersPos(left=100.0, right=0.0)'''
    set_gripper('left', 5)
    set_gripper('right', 5)
    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_tilt"], [-0.7])

    time.sleep(2)
    

def calc_magic(weight_grams, qty_objects):
    magic_number = 8*(weight_grams/100) + qty_objects
    #The magic number represents a 8x8 matrix with the weigth on the left and the elements on the top
    #There are only two blue and four red boxes present so the robot does not lift too much weight!
    switch = {
        9 : "There is one red box in the bag",
        17: "There one blue box in the bag",
        18: "There are two red boxes in the bag",
        26: "There are one red box and one blue box in the bag",
        27: "There are two red boxes and one blue box in the bag",
        34: "There are two blue boxes in the bag",
        35: "There are two red boxes and one blue box in the bag",
        36: "There are four red boxes in the bag",
        43: "There are one red box and two blue boxes in the bag",
        44: "There are three red box and one blue box in the bag",
        52: "There are four red boxes and one blue box in the bag",
        53: "There are two red boxes and two blue boxes in the bag",
        61: "There are three red boxes and two blue boxes in the bag",
        62: "There are four red boxes and two blue boxes in the bag"
    }
    #gets the output accordingly, complains if the input was not correct.
    return switch.get(magic_number, "Something's wrong, I can feel it!")

def center_head_on_object(obj_pos):
    cx, cy = 0.5, 0.5 # Center of image
    obj_x, obj_y = obj_pos

    dist_x = obj_x - cx
    dist_y = obj_y - cy

    head_curr_x = robot.joint_pos["head_pan"]
    head_curr_y = robot.joint_pos["head_tilt"]

    kp = 0.5
    new_head_x = head_curr_x + kp * -dist_x
    new_head_y = head_curr_y + kp * -dist_y

    robot.setJointPos(["head_tilt", "head_pan"], [new_head_y, new_head_x])

effort_values = []
test = []

if __name__ == "__main__":
    
    tickrate = 60
    rate = rospy.Rate(tickrate)
    x_values = []
    y_values_effort = []
    index = count()
    pre_effort = 0 
    global_weight = 0
    first_look_direction = ''
    
    currState = States.INIT
    while not rospy.is_shutdown():

        rate.sleep()

        if robot.buttonCheck("mode"):
            currState = States.INIT

        # if robot.buttonCheck("user"):
        #     currState = States.PICK_BAG

        if DEBUG_MODE:
            cv2.imshow("Image", vision.img_buffer[-1])
            cv2.imshow("Func1", vision.debug_img[0])
            cv2.imshow("Func2", vision.debug_img[1])
            if vision.status[0]:#the first function - detect single color
                print("Area 0: {}".format(vision.results[0][1]))#vision.results[0][0] is the center, vision.result[0][1] is the area.
            if vision.status[1]:#the second function - detect 2 color
                print("Area 1: {}".format(vision.results[1][1]))
            cv2.waitKey(1)

        if currState == States.INIT:
            rospy.loginfo("[INIT]")
            init()
            print("Current Version of Python is: ", sys.version)

            # Transition
            tick_count = 0
            direction = False
            
            finished_cycle = False
            time.sleep(2)
            robot.setJointPos(["head_pan", "head_tilt"], [0.0, 0.75])
            key = raw_input("Press Enter to continue")
            currState = States.FIND_BAG

        elif currState == States.FIND_BAG:
            rospy.loginfo("[FIND BAG]")
            # Move head to find the bag
            head_curr_x = robot.joint_pos["head_pan"]
            if direction == False:
                new_x = head_curr_x + HEAD_SEARCH_SPEED
            else:
                new_x = head_curr_x - HEAD_SEARCH_SPEED

            if new_x > 1.0:
                direction = True
            elif new_x < -1.0:
                direction = False
                finished_cycle = True

            robot.setJointPos(["head_pan", "head_tilt"], [new_x, 0])


        #     # Retrieve results of first function
            status = vision.status[0]

            if -0.7 <= new_x <= -0.6 and status == True and vision.results[0][1] > 1500:
                tick_count += 3
                print("Detected")
                print(tick_count)
                print(new_x)
    

            if tick_count > tickrate // 3:
                tick_count = 0
                currState = States.FOCUS_BAG

        elif currState == States.FOCUS_BAG:
            #print("[FOCUS_BAG]")
            # Retrieve results of first function
            status = vision.status[0]
            if status == True and vision.results[0][1] > 1000:
                tick_count += 3
                center, area = vision.results[0]
                center_head_on_object(center)
                print("[A]")

            if tick_count > tickrate:
                currState = States.FACE_BAG

        elif currState == States.FACE_BAG:
            print("[FACE_BAG]")
            center, area = vision.results[0]
            center_head_on_object(center)
            pan = robot.joint_pos["head_pan"]
            currState = States.PICK_BAG

        elif currState == States.PICK_BAG:
            print("[PICK_BAG]")
            #set_gripper('left', 5)
            #set_gripper('right', 5)
            center, area = vision.results[0]

            # I think this will stop the robot from "jerking" when changing the module
            robot.setGeneralControlModule("none")
            rospy.sleep(1)
            robot.setGeneralControlModule("action_module")
            
            #robot.playMotion(50, wait_for_end=True)

            #get the arm into its starting position
            robot.playMotion(74, wait_for_end=True)
            
            #Bend Knees to support heavier weight
            robot.playMotion(17, wait_for_end=True)

            
            set_gripper('right', 90.0)

            currState = States.WAIT_BAG


        
        elif currState == States.RAISE_RIGHT_HAND:
            print("[RAISE RIGHT HAND]")

            robot.setGeneralControlModule("action_module")
            # robot.playMotion(74, wait_for_end=True)
            # set_gripper('right', 10 )

            r_sho_pitch = robot.joint_effort['r_sho_pitch']
            effort_values.append(r_sho_pitch)

            #TODO: Failsafe for too heavy objects

            if len(effort_values) <= 10:
                avg_r_sho_pitch = np.abs(sum(effort_values) / len(effort_values))
                # print("HIHI")
                print("READY")


            elif len(effort_values) > 10:
                sum_effort = 0

                for i in range(3):
                    current_effort = robot.joint_effort['r_sho_pitch'] - avg_r_sho_pitch
                    # set_gripper('right', 58)

                    # print("HEOIHSE")

                    time.sleep(0.1)

                    robot.playMotion(16, wait_for_end=True)
                    r_sho_pitch = robot.joint_effort['r_sho_pitch']
                    x_values.append(0)
                    y_values_effort.append(np.abs(r_sho_pitch - avg_r_sho_pitch))
                    # print(effort_values)
                    time.sleep(0.1)

                    robot.playMotion(17, wait_for_end=True)
                    r_sho_pitch = robot.joint_effort['r_sho_pitch']
                    x_values.append(1)
                    y_values_effort.append(np.abs(r_sho_pitch - avg_r_sho_pitch))
                    # print(effort_values)
                    time.sleep(0.1)

                    robot.playMotion(18, wait_for_end=True)
                    r_sho_pitch = robot.joint_effort['r_sho_pitch']
                    x_values.append(2)
                    y_values_effort.append(np.abs(r_sho_pitch - avg_r_sho_pitch))
                    # print(effort_values)
                    time.sleep(0.1)

                    robot.playMotion(19, wait_for_end=True)
                    r_sho_pitch = robot.joint_effort['r_sho_pitch']
                    x_values.append(3)
                    y_values_effort.append(np.abs(r_sho_pitch - avg_r_sho_pitch))
                    # print(effort_values)
                    time.sleep(0.1)

                    robot.playMotion(20, wait_for_end=True)
                    r_sho_pitch = robot.joint_effort['r_sho_pitch']
                    x_values.append(4)
                    y_values_effort.append(np.abs(r_sho_pitch - avg_r_sho_pitch))
                    # print(effort_values)
                    time.sleep(3)

                    # set_gripper('right', 10)

                    print(sum(y_values_effort))

                    sum_effort = sum_effort + sum(y_values_effort)
                    
                    y_values_effort[:] = []

                sum_effort = sum_effort / 3
                print(sum_effort)
                # if 1.2 <= sum_effort <= 1.7:
                    # print("The weight is 50 grams")

                if sum_effort <= 1.42:
                    global_weight = 100
                    print("The weight is 100 grams")

                elif sum_effort <= 2.2:
                    global_weight = 200
                    print("The weight is 200 grams")

                elif sum_effort <= 3.2:
                    global_weight = 300
                    print("The weight is 300 grams")

                elif sum_effort <= 4.2:
                    global_weight = 400
                    print("The weight is 400 grams")

                elif sum_effort <= 5.2:
                    global_weight = 500
                    print("The weight is 500 grams")

                elif sum_effort <= 6.2 :
                    global_weight = 600
                    print("The weigth is 600 grams")

                else:
                    global_weight = 700
                    print("The weight is 700 grams or more")

                robot.playMotion(16, wait_for_end=True)

                print("I can sense the intensity of the colors... but I need to make sure..")
                
                currState = States.TALK_USR
        
        elif currState == States.TALK_USR:
            #TODO use TTS for Output and Finger gestures for input
            
            try:
                amount = int(input("How many Boxes are inside the bag? Answer: "))
                print("Okay, now I can feel it..")
                print(calc_magic(global_weight, amount))
                print("TADA!")
                currState = States.REMOVE_BAG
            
            except:
                print("I cannot do much with that answer.. Maybe try once again.")
            
        elif currState == States.INPUT_USR:
            
            print("How many Boxes are inside the bag?"
            )
            done = False

            final_input = -1

            numbers = [6]
            valid_inputs = 0

            while done == False:
                img = vision.img_buffer[-1]
                counter = fcm.FingerCounter

                current_input = counter.count_fingers_frame(img)
                    
                if current_input == 0:
                    numbers[0]+=1
                elif current_input == 1:
                    numbers[1]+=1
                elif current_input == 2:
                    numbers[2]+=1
                elif current_input == 3:
                    numbers[3]+=1
                elif current_input == 4:
                    numbers[4]+=1
                elif current_input == 5:
                    numbers[5]+=1
                else:
                    #No valid input, decrease counter
                    valid_inputs-=1

                valid_inputs+=1

                if valid_inputs > 40:
                    for i in numbers:
                        if i > valid_inputs/0.75:
                            final_input = i
                            done = True

            print(calc_magic(global_weight, amount))
            currState = States.REMOVE_BAG


        elif currState == States.WAIT_BAG:
            print("Waiting for Bag")
            r_sho_pitch = robot.joint_effort['r_sho_pitch']
            
            robot.setJointPos(["head_pan", "head_tilt"], [0.0, 0.75])
            rospy.sleep(1)

            s = 0
            # arithmetic average 
            for i in range(30):
                s = s + robot.joint_effort['r_sho_pitch']
                rospy.sleep(0.02)
            print (s/30)
            if s/30 > 0.49:
                currState = States.RAISE_RIGHT_HAND
            rospy.sleep(1)    #do not start immediately!
            #TODO make robot look at the bag briefly
            # robot.setJointPos(["head_pan", "head_tilt"], [0.5, -0.5])

            

        elif currState == States.REMOVE_BAG:
            robot.setJointPos(["head_pan", "head_tilt"], [0.0, 0.75])
            print("Please remove Bag")
            rospy.sleep(3)
            
            s = 0
            # arithmetic average 
            for i in range(30):
                s = s + robot.joint_effort['r_sho_pitch']
                rospy.sleep(0.02)
            print (s/30)

            #For some reason the effort for holding the bag afterwards is much lower.
            if s/30 < 0.4:
                currState = States.FIND_BAG
            rospy.sleep(1)    #do not start immediately!

            print("Please refill the bag and show it to me")
            
