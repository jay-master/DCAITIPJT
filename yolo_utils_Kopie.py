import numpy as np
import argparse
import cv2
import subprocess
import time
import os
import pprint
import math

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


    """for j in range(len(fbox)):
        fx, fy = fbox[j][0], fbox[j][1]
        fw, fh = fbox[j][2], fbox[j][3]

        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        # text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
        cv2.putText(img, "Free! %d m" % fw, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)"""

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []
    sec1 = []
    sec2 = []
    sec3 = []
    sec4 = []
    sec5 = []
    sec6 = []
    sec7 = []
    sec8 = []
    sec9 = []
    sec10 = []
    #fbox = []

    for out in outs:
        for detection in out:
            # print (detection)
            # a = input('GO!')

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

                """# Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)"""

                ## added part
                h = height  # input size
                w = width

                # sector 1
                if (0.175 * h < centerY < 0.23 * h) and (centerX < 0.124 * w):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec1.append((centerX, centerY, bwidth, bheight))

                # sector 2
                if (0.175 * h < centerY < 0.23 * h) and (0.272 * w < centerX):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec2.append((centerX, centerY, bwidth, bheight))

                # sector 3
                elif (0.228 * h < centerY < 0.29 * h) and (centerX < 0.124 * w):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec3.append((centerX, centerY, bwidth, bheight))

                # sector 4
                elif (0.228 * h < centerY < 0.29 * h) and (0.272 * w < centerX):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec4.append((centerX, centerY, bwidth, bheight))

                # sector 5
                elif (0.377 * h < centerY < 0.456 * h):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec5.append((centerX, centerY, bwidth, bheight))

                # sector 6
                elif (0.614 * h < centerY < 0.719 * h) and (centerX < 0.495 * w):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec6.append((centerX, centerY, bwidth, bheight))

                # sector 7
                elif (0.614 * h < centerY < 0.719 * h) and (0.743 * w < centerX):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec7.append((centerX, centerY, bwidth, bheight))

                # sector 8
                elif (0.763 * h < centerY < 0.842 * h) and (0.743 * w < centerX):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec8.append((centerX, centerY, bwidth, bheight))

                # sector 9
                elif (0.842 * h < centerY < 0.921 * h) and (centerX < 0.495 * w):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec9.append((centerX, centerY, bwidth, bheight))

                # sector 10
                elif (0.94 * h < centerY) and (0.62 * w < centerX):
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

                    sec10.append((centerX, centerY, bwidth, bheight))

    # check free space
    sec1.sort(key=lambda x: x[0])
    sec2.sort(key=lambda x: x[0])
    sec3.sort(key=lambda x: x[0])
    sec4.sort(key=lambda x: x[0])
    sec5.sort(key=lambda x: x[0])
    sec6.sort(key=lambda x: x[0])
    sec7.sort(key=lambda x: x[0])
    sec8.sort(key=lambda x: x[0])
    sec9.sort(key=lambda x: x[0])
    sec10.sort(key=lambda x: x[0])

    conv_ratio_a = 93.97/w # sector 1 ~ 4 [unit: m/pixel]
    conv_ratio_b = 82.11/w # sector 5
    conv_ratio_c = 68.4/w # sector 6, 7
    conv_ratio_d = 58.74/w # sector 8
    conv_ratio_e = 54.97/w # sector 9
    conv_ratio_f = 50.5/w # sector 10

    al1 = math.radians(30)
    al2 = math.radians(45)

    if len(sec1) > 1:
        for i in range(len(sec1)-1):
            thiscar = sec1[i][0]
            nextcar = sec1[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_a

            if actual_disance > 10: # distance between center of bboxes
                parking_lot = actual_disance - 5
                print("Free space in sector 1: %.1f m" % parking_lot)

    if len(sec2) > 1:
        for i in range(len(sec2)-1):
            thiscar = sec2[i][0]
            nextcar = sec2[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_a

            if actual_disance > 10: # distance between center of bboxes
                parking_lot = actual_disance - 5
                print("Free space in sector 2: %.1f m" % parking_lot)

            """if delta_x > 70:

                # free space box
                fx = int(sec2[car_s2][0] + 0.5 * sec2[car_s2][2])
                fy = int(sec2[car_s2][1] + 0.5 * sec2[car_s2][3])
                fw = (sec2[car_s2 + 1][0] - 0.5 * sec2[car_s2 + 1][2]) - (sec2[car_s2][0] + 0.2 * sec2[car_s2][2])
                fh = (sec2[car_s2][3] + sec2[car_s2 + 1][3]) * 0.5

                fbox.append([fx, fy, int(fw), int(fh)])"""

    if len(sec3) > 1:
        for i in range(len(sec3)-1):
            thiscar = sec3[i][0]
            nextcar = sec3[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_a

            if actual_disance > 5.8: # distance between center of bboxes
                parking_lot = actual_disance * math.cos(al1) - 2.5
                print("Free space in sector 3: %.1f m" % parking_lot)

    if len(sec4) > 1:
        for i in range(len(sec4)-1):
            thiscar = sec4[i][0]
            nextcar = sec4[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_a

            if actual_disance > 5.8: # distance between center of bboxes
                parking_lot = actual_disance * math.cos(al1) - 2.5
                print("Free space in sector 4: %.1f m" % parking_lot)

    if len(sec5) > 1:
        for i in range(len(sec5)-1):
            thiscar = sec5[i][0]
            nextcar = sec5[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_b

            if actual_disance > 6.3: # distance between center of bboxes
                parking_lot = actual_disance * math.cos(al2) - 2.5
                print("Free space in sector 5: %.1f m" % parking_lot)

    if len(sec6) > 1:
        for i in range(len(sec6)-1):
            thiscar = sec6[i][0]
            nextcar = sec6[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_c

            if actual_disance > 5.8: # distance between center of bboxes
                parking_lot = actual_disance * math.cos(al1) - 2.5
                print("Free space in sector 6: %.1f m" % parking_lot)

    if len(sec7) > 1:
        for i in range(len(sec7)-1):
            thiscar = sec7[i][0]
            nextcar = sec7[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_c

            if actual_disance > 5.8: # distance between center of bboxes
                parking_lot = actual_disance * math.cos(al1) - 2.5
                print("Free space in sector 7: %.1f m" % parking_lot)

    if len(sec8) > 1:
        for i in range(len(sec8)-1):
            thiscar = sec8[i][0]
            nextcar = sec8[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_d

            if actual_disance > 10: # distance between center of bboxes
                parking_lot = actual_disance - 5
                print("Free space in sector 8: %.1f m" % parking_lot)

    if len(sec9) > 1:
        for i in range(len(sec9)-1):
            thiscar = sec9[i][0]
            nextcar = sec9[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_e

            if actual_disance > 10: # distance between center of bboxes
                parking_lot = actual_disance - 5
                print("Free space in sector 9: %.1f m" % parking_lot)

    if len(sec10) > 1:
        for i in range(len(sec10)-1):
            thiscar = sec10[i][0]
            nextcar = sec10[i + 1][0]
            delta_x = nextcar - thiscar
            actual_disance = delta_x * conv_ratio_f

            if actual_disance > 5: # distance between center of bboxes
                parking_lot = actual_disance - 2.5
                print("Free space in sector 10: %.1f m" % parking_lot)



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
