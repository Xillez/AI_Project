from utils import detector_utils as detector_utils
import copy
import os
import sys
import time

import cv2
import numpy as np

#from pyimagesearch import imutils

payload = None
cap = None

# define range of blue color in HSV
lower_skin = np.array([0, 48, 80])
upper_skin = np.array([20, 255, 255])

face_cascade = None
hand_cascade = None

wd = ""

inputLoc = 0

detection_graph, sess = detector_utils.load_inference_graph()

def programSetup():
    performCommands()

    global wd
    if wd == "":
        wd = os.getcwd()

def performCommands():
    # TODO:
    # - Maybe add save location option?

    global inputLoc

    for i in range(0, len(sys.argv)):
        if sys.argv[i] in ("-c", "--cwd"):
            if os.path.isdir(sys.argv[i + 1]):
                os.chdir(sys.argv[i + 1])
            else:
                print("Couldn't find the specified directory! Defaulting to launch directory!")
        elif sys.argv[i] in ("-i", "--input"):
            if sys.argv[i + 1] != "":
                if sys.argv[i + 1].isdigit():
                    inputLoc = int(sys.argv[i + 1])
                    continue
                inputLoc = sys.argv[i + 1]
        #elif sys.argv[i] in ("<arg_option_1>", "<arg_option_2>"):
        #    action

def displayFrames(titles, frames):
    # Error handling, not all frames assigned colors
    if len(frames) != len(titles):
        print("(len(frames) == len(titles) or len(titles) == 0) not satistfied!")
        return      # Exit

    # Make rectangle in every frame
    for i, frame in enumerate(frames):
        cv2.imshow(titles[i], frame)

def drawFace(facepos, frames, colors = []):
    # Error handling, not all faces assigned colors
    if len(frames) != len(colors) and len(colors) != 0:
        print("(len(frames) == len(colors) or len(colors) == 0) not satistfied!")
        return      # Exit

    # Make rectangle in every frame
    for i, frame in enumerate(frames):
        frame = cv2.rectangle(frame, (facepos[0], facepos[1]),(facepos[0] + facepos[2], facepos[1] + facepos[3]), colors[i] if len(colors) == len(frames) else (255, 255, 255))

class FirstStage:
    def run(self, payload):
        global cap
        global lower_skin
        global upper_skin

        frame = cv2.cvtColor(payload["current_frame"], cv2.COLOR_BGR2HSV) # Save local frame converted to HSV

        frame = cv2.GaussianBlur(frame, (3, 3), 1, 1)           # Blur to help remove some noise

        skinMask = cv2.inRange(frame, lower_skin, upper_skin)   # Find every pixel in skin color range

        skinMask = cv2.medianBlur(skinMask, 5)  # Blur a little more for noise removal

        payload["skin_mask"] = skinMask         # Update current skinmask

class SegmentationStage:
    def run(self, payload):
        global cap
        global face_cascade

        boxes, scores = detector_utils.detect_objects(payload["current_frame"],
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(2, 0.2,
                                         scores, boxes, cap.get(3), cap.get(4),
                                         payload["current_frame"])

        faces = face_cascade.detectMultiScale(payload["current_frame"], 1.5, 5)
        drawFace(faces[0], [payload["current_frame"]])

#def cropToFace(facepos, frame):
#    return cv2.getRectSubPix(frame, (facepos[2], facepos[3]),(facepos[0] + (facepos[2] / 2.0), facepos[1] + (facepos[3] / 2.0)))

def main():
    payload = {}

    # Set up program (set global variable and go through arguments)
    programSetup()

    global cap
    global face_cascade
    #global inputLoc
    # Open video file
    cap = cv2.VideoCapture("./videos/fobi.mp4")
    face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    # Video capturing stream open
    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        # Vaid image, Convert from BGR to RGB
        if ret != False:
            # Does payload contain a "current_frame" key?
            if (len([key for key in payload.keys() if key == "current_frame"]) == 1):
                payload["prev_frame"] = payload["current_frame"]        # Set current frame to prev
            payload["current_frame"] = copy.deepcopy(frame)             # Update current frame
        else:       # Invalid image, start video over, change to be next video in line!
            cap = cv2.VideoCapture("./videos/fobi.mp4")
            continue

        # Run first stage
        sensoryStage = FirstStage()
        sensoryStage.run(payload)
        segmentation = SegmentationStage()
        segmentation.run(payload)

        #grayframe = cv2.cvtColor(payload["current_frame"], cv2.COLOR_BGR2GRAY)
        #hands = hand_cascade.detectMultiScale(grayframe, 1.5, 5)

        #if len(hands) > 0:
        #    self.drawFace(hands[0], [grayframe])# , [(0, 255, 0), (255, 0, 0), (255, 255, 255)])

        #cv2.imshow("hands", grayframe)

        # If any faces, highlight and crop to first face found
        #if len(faces) > 0:
        #    self.drawFace(faces[0], [frame, skinMask])# , [(0, 255, 0), (255, 0, 0), (255, 255, 255)])
        #    #face = self.cropToFace(faces[0], frame)
        #    
        #    # Display all frames with respective titles
        #    #self.displayFrames(["frame", "face", "skin_mask"], [frame, face, skinMask])
        #    self.displayFrames(["frame", "skin_mask"], [frame, skinMask])
        #else:
        
        # Display all frames with respective titles, if avaliable
        # else skip prev_frame until it is avaliable
        if (len([key for key in payload.keys() if key == "prev_frame"]) == 1):
            displayFrames(["current_frame", "prev_frame", "skin_mask"], [payload["current_frame"], payload["prev_frame"], payload["skin_mask"]])
        else:
            displayFrames(["current_frame", "skin_mask"], [payload["current_frame"], payload["skin_mask"]])
        
        # Wait for q press, then exit if pressed
        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    # Release the videocapture
    cap.release()

    # Destroy window
    cv2.destroyAllWindows()

# Main time ot shine
if (__name__ == "__main__"):
    main()

###### Take image and find human colored stuff and mask everything else
#     # define range of blue color in HSV
#     lower_blue = np.array([0, 48, 80])
#     upper_blue = np.array([20, 255, 255])
#     #lower_blue = np.array([0, 0, 0])
#     #upper_blue = np.array([255, 100, 100])
#     #lower_blue = np.array([180, 35, 85])
#     #upper_blue = np.array([240, 95, 85])
#
#     # Threshold the HSV image to get only blue colors
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     mask = cv2.inRange(frame, lower_blue, upper_blue)

#     # apply a series of erosions and dilations to the mask
#     # using an elliptical kernel
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#     mask = cv2.erode(mask, kernel, iterations = 2)
#     mask = cv2.dilate(mask, kernel, iterations = 2)
#
#     # blur the mask to help remove noise
#     mask = cv2.GaussianBlur(mask, (3, 3), 0)
#
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame, frame, mask = mask)
#
#     cv2.imshow("mask", np.hstack([frame, res]))

###### Load image and display, save if "s" pressed
#    img = cv2.imread('./images/DeveloperHeadAche.png',0)
#    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#    cv2.imshow('image',img)
#    k = cv2.waitKey(0)              # if problems with x64 machines, switch with this: k = cv2.waitKey(0) & 0xFF
#    if k == 27:         # wait for ESC key to exit
#        cv2.destroyAllWindows()
#    elif k == ord('s'): # wait for 's' key to save and exit
#        cv2.imwrite('DeveloperHeadAcheGrayscale.png',img)
#        cv2.destroyAllWindows()

###### Capture from default camera, maybe convert to grayscale and save each frame to disk ######
#    cap = cv2.VideoCapture(0)
#
#    while True:
#        # Capture frame by frame
#        ret, frame = cap.read()
#
#        # Convert to grayscale
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#        cv2.imwrite("./images/img_" + str(index) + ".png",gray)
#        # Display resulting image
#        #if grayscaleInput:
#        #    cv2.imshow("frame", gray)
#        #else:
#        #    cv2.imshow("frame", frame)
#
#        cv2.imshow("frame", frame if not grayscaleInput else gray)
#        index += 1
#
#        # Wait for q press, then exit if pressed
#        if cv2.waitKey(1) & 0xff == ord("q"):
#            break
#    
#    # Release resource
#    cap.release()
#
#    # Destroy window
#    cv2.destroyAllWindows()
