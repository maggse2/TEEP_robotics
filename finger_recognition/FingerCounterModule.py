"""Module to count fingers as an input Method for a Robot. Original idea by Murtaza Hassan,
    Modifications by Max Paulenz"""

import cv2
import time
import HandTrackingModule as htm


class FingerCounter:
    def __init__(self):
        self.w_cam, self.h_cam = 640, 480

        self.previous_time = 0

        self.hand_detector = htm.HandDetector(min_detection_confidence=0.75, max_num_hands=1)

        self.tip_ids = [4, 8, 12, 16, 20]

    def count_fingers_frame(self, img):
        img = self.hand_detector.find_hands(img)
        landmark_list = self.hand_detector.find_position(img, draw=False)
        if len(landmark_list) != 0:
            fingers = []
            # Thumb
            if landmark_list[0][1] > landmark_list[1][1]:
                # print("Back of Hand")
                if landmark_list[self.tip_ids[0]][1] < landmark_list[self.tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # print("Front of Hand")
                if landmark_list[self.tip_ids[0]][1] > landmark_list[self.tip_ids[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if landmark_list[self.tip_ids[id]][2] < landmark_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # print(fingers)

            total_fingers = fingers.count(1)
            print(total_fingers, " fingers detected in frame")
            return total_fingers

    def catch_input(self):
        cap = cv2.VideoCapture(0)
        cap.set(2, self.w_cam)
        cap.set(4, self.h_cam)

        print("How many Boxes are inside the bag?")
        done = False

        final_input = -1

        numbers = []
        for i in range(6):
            numbers.append(0)
        valid_inputs = 0

        while done == False:
            success, img = cap.read()

            current_input = self.count_fingers_frame(img)

            if current_input == 0:
                numbers[0] += 1
            elif current_input == 1:
                numbers[1] += 1
            elif current_input == 2:
                numbers[2] += 1
            elif current_input == 3:
                numbers[3] += 1
            elif current_input == 4:
                numbers[4] += 1
            elif current_input == 5:
                numbers[5] += 1
            else:
                # No valid input, decrease counter
                valid_inputs -= 1

            cv2.imshow("Image", img)
            cv2.waitKey(1)

            valid_inputs += 1
            print(valid_inputs, " valid inputs")

            if valid_inputs > 40:
                counter = -1
                for i in numbers:
                    counter += 1
                    print("Current Number:", counter, "With ", i," Instances")
                    if i > (valid_inputs * 0.65):
                        final_input = counter
                        done = True
                        cv2.putText(img, str(int(final_input)), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 5)
                        while True:
                            cv2.imshow("Image", img)
                            cv2.waitKey(1)


        print("Final input:", final_input)





    def visual_debugger(self):
        # Sets up the Video Capture, in the robot, a video capture will be given by th node
        cap = cv2.VideoCapture(0)
        cap.set(2, self.w_cam)
        cap.set(4, self.h_cam)
        while True:
            success, img = cap.read()
            img = self.hand_detector.find_hands(img)
            landmark_list = self.hand_detector.find_position(img, draw=False)

            print(landmark_list)

            if len(landmark_list) != 0:
                fingers = []
                # Thumb
                if landmark_list[0][1] > landmark_list[1][1]:
                    print("Back of Hand")
                    if landmark_list[self.tip_ids[0]][1] < landmark_list[self.tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    print("Front of Hand")
                    if landmark_list[self.tip_ids[0]][1] > landmark_list[self.tip_ids[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # 4 Fingers
                for id in range(1, 5):
                    if landmark_list[self.tip_ids[id]][2] < landmark_list[self.tip_ids[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    # print(fingers)

                total_fingers = fingers.count(1)
                print(total_fingers)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


def main():
    finger_counter = FingerCounter()
    finger_counter.catch_input()


if __name__ == "__main__":
    main()
