import cv2
import argparse
import numpy as np
import glob

class ObjectDetection:
    def __init__(self, classFile='OD/configurationFiles/Classes.txt',weightsFile='OD/configurationFiles/yolov3.weights', configFile='OD/configurationFiles/yolov3.cfg.txt'):
        self.classes = None
        with open(classFile, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(weightsFile, configFile)


    def createBlob(self, image):
        width = image.shape[1]
        height = image.shape[0]
        scale = 0.00392

        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        self.net.setInput(blob)


    # function to get the output layer names
    # in the architecture
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers


    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.colors[class_id]
        img = cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        return cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detectObjects(self, image):
        ol = self.get_output_layers()
        self.createBlob(image)
        out_layers = self.net.forward(ol)
        classIDS = []
        confidence_list = []
        packs = []

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for output_layer in out_layers:
            for detection in output_layer:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:   #Putting the confidence threshold as 0.5
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    classIDS.append(classID)
                    confidence_list.append(float(confidence))
                    packs.append([x, y, w, h])

        # apply non-max suppression
        conf_threshold = 0.5
        nmaxs_threshold = 0.4
        indices = cv2.dnn.NMSBoxes(packs, confidence_list, conf_threshold, nmaxs_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            box = packs[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_bounding_box(image, classIDS[i], confidence_list[i], round(x), round(y), round(x + w), round(y + h))
        return image