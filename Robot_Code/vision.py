import rospy
import cv2
import numpy as np
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class VisionSystem:
    """
        To be used as a callback for a Subscriber to a camera topic, saves
        the images to a limited buffer. Can also run a sequence of functions
        on the image as soon as it is captured. Each function in the pipeline
        should return a tuple of its resulting value and success status. The
        first argument of the function should be an image.

        foo(img, *args) -> (result, success)

        Parameters:
            maxlen: The maximum size of the image buffer, old images are
                discarded.
            pipeline_funcs: A list of functions to be ran after reading a new
                image.
            pipeline_args: The arguments to each of the functions.

    """
    def __init__(self, maxlen=1, pipeline_funcs=[], pipeline_args=[], debug=False, verbose=1):
        self.verbose = verbose
        self.frame_count = 0
        self.img_buffer = deque(maxlen=maxlen)
        self.bridge = CvBridge()

        self.pipeline_funcs = pipeline_funcs
        self.pipeline_args = pipeline_args

        self.results = [None] * len(pipeline_funcs)
        self.status = [None] * len(pipeline_funcs)
        self.debug_img = [None] * len(pipeline_funcs)

        self.debug = debug

    def read(self, ros_msg=None):
        """ Acquires a new frame from a ROS message. This function is intended to
            be passed as callback when subscribing to a camera topic.

            Parameters:
                ros_msg: A ros message containing the image
        
        """
        # print('im read')
        try:
            
            img = self.bridge.imgmsg_to_cv2(ros_msg, "bgr8")
            cv2.imwrite('/home/robotis/joe_ws/src/fira_basketball/config/output.jpg', img)
            # print(img)
            self.img_buffer.append(img)
            self.frame_count += 1
        except err:
            rospy.loginfo(err)

        i_foo = 0
        # print(self.pipeline_funcs, self.pipeline_args)
        for func, args in zip(self.pipeline_funcs, self.pipeline_args):
            try:
                # The image passed is a copy so further functions are not affected
                copy_img = img.copy()
                result, success = func(copy_img, *args)
                if self.debug:
                    self.debug_img[i_foo] = copy_img

                self.status[i_foo] = success
                if success != False:
                    self.results[i_foo] = result

            except Exception as e:
                if self.verbose == 1:
                    rospy.loginfo("Failed to run function %d in vision pipeline" % i_foo)
                    rospy.loginfo(e)
                self.status[i_foo] = False

            i_foo += 1

def detect2Color(img, hsv_params1, hsv_params2):
    """ Detects a single color specified in HSV colorspace in the img_buffer
        
        Parameters:
            img 
            hsv_params: A tuple of two 3-dim numpy array specifing the HSV 
                range of the color.

        Return:
            (pos_x, pos_y): Returns the center position of the largest color
                blob normalized to the dimensions of the image.
            area: area of the largest color blob

    """
    lower1 = hsv_params1[0]
    upper1 = hsv_params1[1]
    lower2 = hsv_params2[0]
    upper2 = hsv_params2[1]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5, 5)))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((5, 5)))

    mask = cv2.bitwise_or(mask1, mask2)

    center_pos, area = findCenterOfLargestContour(mask)

    img[:, :, 2] = mask
    img[:, :, :2] = 0

    if center_pos is not None:
        # Normalize by image dimension
        c_x = center_pos[0] / float(img.shape[1])
        c_y = center_pos[1] / float(img.shape[0])

        center = (c_x, c_y)
        
        return (center, area), True
    else:
        return None, False

def get_contour_areas(contours):
    all_area = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_area.append(area)

    return all_area

def detectSingleColor(img, hsv_params):
    '''
    lower = hsv_params[0]
    upper = hsv_params[1]
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)

    mask = cv2.inRange(hsv_img, lower, upper)
    
    #element = cv2.getStructuringElement(cv2.MORPH_RECT(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    
    ret, thresh = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY)
    #_, contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    get_contour_areas(contours)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    if len(sorted_contours) != 0:
        largest_item = sorted_contours[0]
        cv2.drawContours(image = img, contours = [largest_item], contourIdx = -1, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
        x = 0
        y = 0
    
        for i in range (0, len(largest_item)):
            x = x + largest_item[i][0][0]
            y = y + largest_item[i][0][1]

        if contours is not None:
            c_x = int(x / len(largest_item))
            c_y = int(y / len(largest_item))

            center = (c_x, c_y)
        area = cv2.contourArea(largest_item)
        #cv2.imshow('img', img)
        #cv2.waitKey(10)
        print('in')
        return (center, area), True
    else:
        #cv2.imshow('img2', img)
        #cv2.waitKey(1)
        return None, False


    
    Detects a single color specified in HSV colorspace in the img_buffer
        
        Parameters:
            img 
            hsv_params: A tuple of two 3-dim numpy array specifing the HSV 
                range of the color.

        Return:
            (pos_x, pos_y): Returns the center position of the largest color
                blob normalized to the dimensions of the image.
            area: area of the largest color blob
    '''
    
    lower = hsv_params[0]
    upper = hsv_params[1]

    # print(hsv_params)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

    center_pos, area = findCenterOfLargestContour(mask)

    img[:, :, 2] = mask
    img[:, :, :2] = 0
    
    if center_pos is not None:
        # Normalize by image dimension
        c_x = center_pos[0] / float(img.shape[1])
        c_y = center_pos[1] / float(img.shape[0])

        center = (c_x, c_y)
        
        return (center, area), True
    else:
        return None, False
    
def findCenterOfLargestContour(binary_mask):
    """ Detects all contours in the image and returns the center position and 
        area of the largest contour.

        Parameters:
            binary_mask: A binary image, to detect the contours.

        Returns:
            (center_x, center_y), area: If no contours are detect it returns
                None, None.

    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 1:
        largest_cnt = 0
        largest_area = cv2.contourArea(contours[0])
    elif len(contours) > 1:
        largest_cnt = 0
        largest_area = cv2.contourArea(contours[0])
        for i, cnt in enumerate(contours[1:]):
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > largest_area:
                largest_area = cnt_area
                largest_cnt = i+1 # Enumerate starts from 0, increment 1 here 
                                  # because we skip the first contour
    else: # No contours were found
        return None, None

    # Get moments of largest contour
    M = cv2.moments(contours[largest_cnt])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy), largest_area
