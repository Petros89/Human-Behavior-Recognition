###          Petros Apostolou | Jan 2021               ###
##########################################################
###   This file is a copyright of python3 program      ###
###   --------------------------------------------     ###
###   Human Behavior Recognition Pipeline              ###
###   Includes:                                        ###
###   Video streaming using opencv FileVideoStream     ###
###   Face detection and landmark keypoints            ###
###   3D head pose detection and transformation        ###
###   Eye Aspect Ratio Estimation for Eye-Blinking     ###
###   Eye-Tic recognition using deque with MaxLength   ###
###   Author: Petros Apostolou                         ###
###   Email: apost035@umn.edu                          ###
###   Last Update: 23-8-2021                           ###
##########################################################

### Facial Detection Using Video Streaming (dlib,facial_utils,face_shape_predictor)


# import the necessary packages
from collections import deque
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import datetime
import dlib
import cv2
import headpose
import os



def anonymize_face_simple(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)
        # ensure the width of the kernel is odd
        if kW % 2 == 0:
        	kW -= 1
        # ensure the height of the kernel is odd
        if kH % 2 == 0:
	        kH -= 1
        # apply a Gaussian blur to the input image using our computed
        # kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks):
	# divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
	        for j in range(1, len(xSteps)):
		        # compute the starting and ending (x, y)-coordinates
		        # for the current block
		        startX = xSteps[j - 1]
		        startY = ySteps[i - 1]
		        endX = xSteps[j]
		        endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
		        roi = image[startY:endY, startX:endX]
		        (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
		        cv2.rectangle(image, (startX, startY), (endX, endY),
			        (B, G, R), -1)
	# return the pixelated blurred image
        return image




def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

### def facial_landmarks(FACIAL_LANDMARKS_IDXS):
###         # define a dictionary that maps the indexes of the facial # landmarks to specific face regions
###         FACIAL_LANDMARKS_IDXS = OrderedDict([
### 	        ("mouth", (48, 68)),
### 	        ("right_eyebrow", (17, 22)),
### 	        ("left_eyebrow", (22, 27)),
### 	        ("right_eye", (36, 42)),
### 	        ("left_eye", (42, 48)),
### 	        ("nose", (27, 35)),
### 	        ("jaw", (0, 17))
###         ])
### 
###         return FACIAL_LANDMARKS_IDXS

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    eye_ratio = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return eye_ratio


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

## USE STORED VIDEO STREAM ###
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
#time.sleep(1.0) 
fps = FPS().start()
#

## USE WEB CAMERA ###
## initialize the video stream and allow the cammera sensor to warmup
#print("[INFO] camera sensor warming up...")

# USE WEBCAM
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
#fileStream = False
#fps = FPS().start()

## Initialize Setting for Eye Tracking ##
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
COUNTER_BLINK = 0
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNT_TIC=0
EYE_TIC_CONSEC_FRAMES = 12
COUNTER_TIC = 0
EYE_BLINK = 0
EYE_TIC = 0

## ---------------------------------------------------------- ##
# grab the indexes of the facial landmarks 

# grab the indexes of the facial landmarks for the jaw
#(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# grab the indexes of the facial landmarks for the eye brows
#(rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
#(lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

# grab the indexes of the facial landmarks for the eyes
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

# grab the indexes of the facial landmarks for the nose
#(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# grab the indexes of the facial landmarks for the mouth
#(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
## ---------------------------------------------------------- ##
##################################################################
firstFrame = None
# Reset timer
timer = 0.0
t = 0.0001


###------------------FacialRecognitionStreaming---------------------------------###
#==> Insert head pose estimator
hpd = headpose.HeadposeDetection(1,"shape_predictor_68_face_landmarks.dat")

# some initializations
head_tics = 0
flag1 = 0
flag2 = 0
flag3 = 0
count=0
img_cnt=0
 
### behavioral recognition constants ###
# size of the localized vector
VEC_SIZE = 5
# deque of frames
FRAMES = deque(maxlen=VEC_SIZE)

while True:

    # Implement Time Tracker
    # Add timer for tracking streaming duration
    # check if the writer is None
    
    (mins, secs) = divmod(t, 60)
    mins = 6.5*mins
    secs = 6.5*secs
    mins=int(mins)
    timer = '{:02d}:{:.2f}'.format(mins, secs)
    #print(timer, end="\r") 
    #time.sleep(0.01) 
    t += 0.01
    #----------------------------------------------------------------



    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
    	break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)

    # start frame streaming
    frame = vs.read()

    # resize image
    #frame = cv2.resize(frame, (450,450))#, interpolation = cv2.INTER_AREA)
    if type(frame) != np.ndarray:
        key = ord("q")
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        print("[INFO] Total Eye Blinks: {}".format(EYE_BLINK))
        print("[INFO] Total Eye Tics: {}".format(EYE_TIC))
        print("[INFO] Total Head Tics: {}".format(head_tics))
        print("[INFO] Total Motor Tics: {}".format(total_tics))
        break

    frame = imutils.resize(frame, width=450)
    
    # if our queue is not filled to the vector size, continue back to
    # # the top of the loop and continue polling/processing frames

    # now that our frames array is filled we can construct our blob
    #blob = cv2.dnn.blobFromImages(FRAMES, 1.0, (FRAME_SIZE, FRAME_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    #blob = np.transpose(blob, (1, 0, 2, 3))
    #blob = np.expand_dims(blob, axis=0)



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray)

    # get rotation angles from class headpose
    res, angles = hpd.process_image(frame)

    

    # loop over the face detections
    #for rect in rects:
    for (i, rect) in enumerate(rects):


        # draw rectangle box around face
        (x, y, w, h) = rect_to_bb(rect)

        # pass Gaussian filter to pixelate video for anonymizing it
        #face = anonymize_face_pixelate(frame, 10)
        # Select only detected face portion for Blur
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract all landmarks coordinates
        #jaw = shape[jStart:jEnd]
        #rightEyeBrow = shape[rbStart:rbEnd]
        #leftEyeBrow = shape[lbStart:lbEnd]
        rightEye = shape[rStart:rEnd]
        leftEye = shape[lStart:lEnd]
        #nose = shape[nStart:nEnd]
        #mouth = shape[mStart:mEnd]


        # Get some motion detection
        # ---------------------------------------------------------------------
        # compute the absolute difference between the current frame and
        # first frame
        #if firstFrame is None:
        #    firstFrame = gray
        #    continue
        #frameDelta = cv2.absdiff(firstFrame, gray)
        #thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        #thresh = cv2.dilate(thresh, None, iterations=2)
        #cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
         
        # loop over the contours
        #for c in cnts:
        #    # if the contour is too small, ignore it
        #    if cv2.contourArea(c) < args["min_area"]:
        #        continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
            #(x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #text = "Occupied"
        #cv2.imshow("Motion Detector", frameDelta)
            #if np.mean(np.square(c) > args["min_area"]:
            #    print("Some Motion")
        #-----------------------------------------------------------------------
        
        # extract the mouth coordinates
        # visualize mouth
        #jawHull = cv2.convexHull(jaw)
        #leftEyeBrowHull = cv2.convexHull(leftEyeBrow)
        #rightEyeBrowHull = cv2.convexHull(rightEyeBrow)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #noseHull = cv2.convexHull(nose)
        #mouthHull = cv2.convexHull(mouth)
        

         
        # compute and visualize the convex hull for all landmarks
        
        # JAW
        #cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)

        # EYES BROWS
        #cv2.drawContours(frame, [leftEyeBrowHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeBrowHull], -1, (0, 255, 0), 1)

        # EYES
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # MOUTH
        #cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        # NOSE
        #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        

        # Perform EYE_TRACKING   
        # EYE CHECK 
        # count eye motion
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        eye_ratio = (leftEAR + rightEAR) / 2.0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if eye_ratio <= EYE_AR_THRESH:
            COUNTER_BLINK += 1
            COUNTER_TIC +=1
        # ot3herwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES:
                EYE_BLINK += 1
                # append consecutive frames to the localized vector
                FRAMES.append(frame)
            COUNTER_BLINK = 0
           

            if len(FRAMES) >= VEC_SIZE and COUNTER_TIC >= EYE_TIC_CONSEC_FRAMES:
                EYE_TIC +=1
        
                
            # reset the eye frame counter
            COUNTER_TIC = 0


        # Eye-Blink Tracking
        cv2.putText(frame, "Eye Blinks: {}".format(EYE_BLINK),(280,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)

        # Eye-Tic Tracking
        cv2.putText(frame, "Eye Tics: {}".format(EYE_TIC), (280,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Aspect Ratio Tracking
        cv2.putText(frame, "EAR: {:.1f}".format(eye_ratio), (280,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Timer [secs] Tracking
        #cv2.putText(frame, "Timer[sec]: {}".format(timer),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)

        # Euler angles [Rx, Ry, Rz] Tracking
        if type(angles) is list:
            cv2.putText(frame, "R(x): %+06.2f" % angles[0], (0,45),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0),2)
            cv2.putText(frame, "R(y): %+06.2f" % angles[1], (0,65),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0),2)
            cv2.putText(frame, "R(z): %+06.2f" % angles[2], (0,85),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0),2)


            # Head Tic Detection Based on Euler Angles Thresholds and smart flags
            #cv2.putText(self.im, "rot(y): %+06.2f" % angles[1], (px, (4*py + dy)), font,fontScale=fs, color=fontColor, thickness=2)
            #cv2.putText(self.im, "rot(z): %+06.2f" % angles[0], (px, (4*py + 2*dy)), font,fontScale=fs, color=fontColor, thickness=2)
            if angles[0] >= 10.0 and flag1 == 0:
                cv2.putText(frame, "Head-Tic: rot = %+06.2f degs" %angles[0],(20,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
                head_tics+=1
                flag1 = 1
            else:
                flag1 = 0
            if np.abs(angles[1]) >= 30.0 and flag2 == 0:
                cv2.putText(frame, "Head-Tic: rot = %+06.2f degs" %angles[1],(20,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
                head_tics+=1
                flag2 = 1
            else:
                flag2 = 0
            if np.abs(angles[2]) >= 30.0 and flag3 == 0:
                cv2.putText(frame, "Head-Tic: rot = %+06.2f degs" %angles[2],(20,150),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)
                head_tics+=1
            else:
                flag3 = 0



        # compute total motor tics (head + eye tics)
        total_tics = head_tics + EYE_TIC


        # Head Tics Tracking
        cv2.putText(frame, "Head Tics: {}".format(head_tics),(0,180),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)

        # Total Motor Tics Tracking (Head + Eye Tics)    
        cv2.putText(frame, "Total Motor Tics: {}".format(total_tics),(0,210),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,0), 2)



    if type(angles) is not list:
        # exlude frames with occlusions
        #cv2.imshow("Occlusions Observed", frame)
        pass
    else:
        # show the frame
        cv2.imshow("Facial Detector", frame)

        # write frames stdout
        #cv2.imwrite('EyeTics/petros'+str(img_cnt)+'.jpg',frame)
        #img_cnt+=1
        #pass


    key = cv2.waitKey(1) & 0xFF
    #fps.update()


    # save frames per fps (0.1)
    #cv2.imwrite("frames/frame{:f}.jpg".format(count), frame)
    #count+=0.1
        
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        break

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] Total Eye Blinks: {}".format(EYE_BLINK))
print("[INFO] Total Eye Tics: {}".format(EYE_TIC))
print("[INFO] Total Head Tics: {}".format(head_tics))
print("[INFO] Total Motor Tics: {}".format(total_tics))


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
exit()
