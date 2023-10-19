import cv2
import numpy as np
from random import choice
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import sys
from time import time,sleep
sys.path.append(r'C:\Users\SHUBH MEHTA\Documents\Pycharm Projects')
from Custom_Hands import custom as ch


labels = {0:'I',
          1:'L',
          2:'O',
          3:'V',
          4:'E',
          5:'Y',
          6:'U',
          7:'restart'}


padding = 20
image_size = 400
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
capture_started = False
capture_seconds = 3
result_seconds=3
sentence_seconds=5
sentence=""

def draw_box(frame, color, length=50, thickness=2):
    """
    Draws a box with specified characteristics on the given frame.

    Args:
        frame (numpy.ndarray): The image frame on which the box will be drawn.
        color (tuple): The color of the box lines in (B, G, R) format.
        length (int, optional): The length of each line segment of the box. Default is 50.
        thickness (int, optional): The thickness of the box lines. Default is 2.
    """
    # Define corner points for the box
    x1, y1 = 320, 10
    x2, y2 = 630, 10
    x3, y3 = 320, 300
    x4, y4 = 630, 300

    # Draw lines to create the box
    cv2.line(frame, (x1, y1), (x1 + length, y1), color=color, thickness=thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color=color, thickness=thickness)

    cv2.line(frame, (x2, y2), (x2 - length, y2), color=color, thickness=thickness)
    cv2.line(frame, (x2, y2), (x2, y2 + length), color=color, thickness=thickness)

    cv2.line(frame, (x3, y3), (x3 + length, y3), color=color, thickness=thickness)
    cv2.line(frame, (x3, y3), (x3, y3 - length), color=color, thickness=thickness)

    cv2.line(frame, (x4, y4), (x4 - length, y4), color=color, thickness=thickness)
    cv2.line(frame, (x4, y4), (x4, y4 - length), color=color, thickness=thickness)

while True:
    success,frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame,1)
    draw_box(frame, (0, 0, 255))
    hands,annotated_frame = detector.findHands(frame, flipType=False, draw=False)
    frame = ch(frame, hands, 4, 2, 4, 2, padding)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        crop = frame[y - padding:y + h + padding, x - padding:x + w + padding]
        white = np.ones((image_size, image_size, 3), np.uint8) * 255
        height, width = crop.shape[0], crop.shape[1]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            white = np.ones((image_size, image_size, 3), np.uint8) * 255
            height, width = crop.shape[0], crop.shape[1]
            if height / width > 1:
                new_height = image_size
                new_width = int((image_size / height) * width)
            else:
                new_width = image_size
                new_height = int((image_size / width) * height)

            if new_width > 0 and new_height > 0:
                new_image = cv2.resize(crop, (new_width, new_height))
                gap_x = int((image_size - new_width) / 2)
                gap_y = int((image_size - new_height) / 2)
                white[gap_y:gap_y + new_height, gap_x:gap_x + new_width] = new_image
            else:
                print("Invalid dimensions for resizing")

            if (x >= 320) and (x + w <= 630) and (y >= 10) and (y + h <= 300):
                draw_box(frame, (0, 255, 0))

                if not capture_started:
                    capture_start_time = time()
                    capture_started=True
                else:
                    elapsed_capture_time = int(time()-capture_start_time)
                    remaining_capture_time = max(0,capture_seconds-int(elapsed_capture_time))
                    cv2.putText(frame, f"Capturing in : {remaining_capture_time}", (10, 400), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 255), 2)

                    if remaining_capture_time==0:
                        pred, index = Classifier.getPrediction(classifier, white, draw=False)
                        letter = labels[np.argmax(pred)]
                        if letter == 'restart':
                            cv2.namedWindow('Sentence')
                            sentence_screen_start_time = time()
                            while int(time()-sentence_screen_start_time)<sentence_seconds:
                                sentence_screen = np.ones((480, 640, 3), np.uint8) * 255
                                cv2.putText(sentence_screen, sentence, (20, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 1)
                                cv2.putText(sentence_screen, 'New Capture starts in :', (20,300), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 0), 1)
                                cv2.putText(sentence_screen, f'{sentence_seconds - int(time() - sentence_screen_start_time)}', (20,400),cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 0), 1)
                                cv2.imshow('Sentence', sentence_screen)
                                cv2.waitKey(1)
                            capture_started = False
                            sentence=""
                            cv2.destroyWindow('Sentence')
                        else:
                            result_start_time = time()
                            cv2.namedWindow('Result')
                            #cv2.resizeWindow('Result',1200,800)
                            while int(time()-result_start_time)<result_seconds:
                                result = cv2.imread(f'alphabets/{letter}.jpeg')
                                result = cv2.resize(result,(640,480))
                                cv2.imshow('Result',result)
                                cv2.waitKey(1)
                            sentence = sentence+letter
                            capture_started=False
                            cv2.destroyWindow('Result')
            else:
                capture_started=False
                cv2.putText(frame, '-->Please Place your hand ', (10, 350), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 1)
                cv2.putText(frame, 'inside the box', (80, 380), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 1)

        else:
            capture_started=False
            cv2.putText(frame, '-->Please Place your hand ', (10, 350), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 1)
            cv2.putText(frame, 'inside the box', (80, 380), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 1)
    else:
        capture_started=False
        cv2.putText(frame, '-->Please Place your hand ', (10, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, 'inside the box', (80, 380), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('Screen',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

