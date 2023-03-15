# TEEP_robotics
This Repository is for all the python scripts I developed during my September 2022 Internship at National Taiwan Normal University.

The task was to make a ROBOTIS OP3 robot perform a magic show in front of an audience without giving away the inner workings of his tricks.
I transformed pre-existing code in a way to make the robot accept a bag and weigh its contents in order to determine the color of boxes inside said bag.

Boxes of the same color will contain equal weights to make them distinguishable for the robot. The Boxes are provided in such quantities that the robot can use a truth table to unequivocally deduct the color of the boxes, given the information of how many boxes are inside the bag.

My initial intent was to have the robot read the number of boxes from the participants hand and also receive commands via handsign inputs from its front facing camera.
After developing the hand tracking module using Google's mediapipe library in Python, I realized that the existing code could not run on Python3 making it incompatible with my newly developed hand tracking module.
Due to time constraints on my one month intership, the modules remained seperate.

Robot_Code contains all code used to make the robot perform the magic trick of guessing the contents.
Assignments contains my solution to an assignment on multi armed bandit optimization from Prof. Saeeds Reinforcement Leanring lecture, I attended at NTNU.

