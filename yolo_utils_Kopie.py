import numpy as np
import argparse
import cv2
import subprocess
import time
import os
import pprint

def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            #######################################################################################
            # added
            #print('box coordinate of detected vehicle: ', x, y, x+w, y+h)
            #print('center of bbox: ', (x+x+w)/2, (y+y+h)/2)
            #######################################################################################
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            #text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            text = "{}".format(labels[classids[i]])
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)




    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []
    cars = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list: center coordinates for distance calculation

                h = height
                w = width
                ## added part
                # Append to list
                if (0.175*h < centerY < 0.29*h and (centerX < 0.124*w or 0.248*w < centerX)) or (0.377*h < centerY < 0.456*h) or (0.614*h < centerY < 0.79*h and (centerX < 0.495*w or 0.693*w < centerX)) or (centerY > 0.842*h and (centerX < 0.495*w or 0.62*w < centerX)):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    cars.append([centerX, centerY])

                    #print('center of bbox: ', centerX, centerY)

    #print(cars)
    sorted(cars, )

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
