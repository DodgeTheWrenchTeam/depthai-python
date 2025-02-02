#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
sys.path.append('../../../')
from DodgeTheWrench.Avoidance import DodgeWrench
from DodgeTheWrench.MoveMotor import MoveMotor
import RPi.GPIO as GPIO

# Set up LEDs
greenLED = 27
redLED = 17
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Initialize GPIO pins
GPIO.setup(greenLED, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(redLED, GPIO.OUT, initial=GPIO.LOW)

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('../models/custom_yolo/416_416_yolo_v4_tiny_openvino_2021.3_6shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnBlobPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Tiny yolo v3/4 label texts
labelMap = ["Tennis Ball"]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(416,416)
vidx = int(416*1)
vidy = int(416*1)
#camRgb.setVideoSize(vidx, vidy)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# This value was used during calibration
camRgb.initialControl.setManualFocus(130)
camRgb.setIspScale(2, 3)
#camRgb.setFps(30)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
#monoLeft.setFps(30)
#monoRight.setFps(30)
# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.7)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.1)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(10000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(1)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Initialize MoveMotor, home linear actuator
move = MoveMotor()
move.home(150)
fpscount = 0
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMappingQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    positionList = []
    currentPosition = []
    
    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections
        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMappingQueue.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        
        
        GPIO.output(greenLED,GPIO.HIGH)
        GPIO.output(redLED,GPIO.LOW)

        

        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            
            
            # Updating position list as Oak-D tracks tennis ball
            currentPosition = [detection.spatialCoordinates.x, detection.spatialCoordinates.y, detection.spatialCoordinates.z] 
            
            # Continuing to next loop if no object is picked up
            if currentPosition == [0.0, 0.0, 0.0]:
                continue
            
            # Increasing sample count for each tracked position
            # Only append the current position if the ball is within 1500 cm
            #if currentPosition[2] < 1500:
            positionList.append(currentPosition)
#             if len(positionList) == 1:
#                 start = time.time()
#             if len(positionList) == 101:
                #print(time.time() - start)
            
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        # Upon having six recorded positions, start recording an initial and final point for determining approximate velocity
        while len(positionList) > 1:
            position1 = positionList[0]
            position2 = positionList[1]

            # Removing oldest position to renew initial and final points as ball travels
            positionList = []
            # Could also use pop here, need to test which is better
            #positionList.pop(0)
            
        # Running avoidance algorithm with two positions, third argument is tolerance for avoidance in mm
            if position2[2] < 2000:
                dirMove, moveDist = DodgeWrench(position1, position2, 300, 1500, 20, 1)
        
                if dirMove != "Stay":
                    #print(position1, position2)
                    if dirMove == "Move Either Way":
                            dirMove = "right"
                    #print('Move', moveDist, 'mm to the', dirMove)
                #else:
                    #print(dirMove)
                
                # Moving the motor depending on the command result
                if moveDist < 150:
                    moveDist = 150
                if (dirMove == "right"):
                    #move.moveMotor("right",1000,200)
                    move.accelerate("right",10,40,900,moveDist)
                    GPIO.output(greenLED,GPIO.LOW)
                    GPIO.output(redLED,GPIO.HIGH)

                    time.sleep(1)
                    #move.moveMotor("left",1000,200)
                    move.accelerate("left",40,40,900,moveDist)
                    time.sleep(1.5)
                    #GPIO.output(greenLED,GPIO.HIGH)

                elif (dirMove == "left"):
                    #move.moveMotor("left",1000,200)
                    move.accelerate("left",10,40,900,moveDist)
                    GPIO.output(greenLED,GPIO.LOW)
                    GPIO.output(redLED,GPIO.HIGH)
                    time.sleep(1)
                    #move.moveMotor("right",1000,200)
                    move.accelerate("right",40,40,900,moveDist)
                    time.sleep(1.5)
                    #GPIO.output(greenLED,GPIO.HIGH)
                    #GPIO.output(redLED,GPIO.LOW)
        
        
        
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        fpscount +=1
        if fpscount == 30:
            print(fps)
            fpscount = 0
        #cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
