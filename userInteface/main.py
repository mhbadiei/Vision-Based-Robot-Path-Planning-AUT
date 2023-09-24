import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

# Define Variables
MQTT_BROKER = "192.168.43.115"
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 45
MQTT_TOPIC = "testTopic"


# Define on_connect event Handler
def on_connect(self,mosq, obj, rc):
    #Subscribe to a the Topic
    mqttc.subscribe(MQTT_TOPIC, 0)

# Define on_subscribe event Handler
def on_subscribe(mosq, obj, mid, granted_qos):
    print("Subscribed to MQTT Topic")

# Define on_message event Handler
def on_message(mosq, obj, msg):
    print(msg.payload)


bg = None
confidence_array = [0] * 100
x_data = list(range(100))
confidence_counter = 0
flag = False
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    start_recording = False

    # Initiate MQTT Client                
    mqttc = mqtt.Client()

    # Register Event Handlers
    mqttc.on_message = on_message
    mqttc.on_connect = on_connect
    mqttc.on_subscribe = on_subscribe
    # Connect with MQTT Broker
    mqttc.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL )
    
    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width = 700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                    # Publish message to MQTT Topic 
                    #mqttc.publish(MQTT_TOPIC,predictedClass)
                    if predictedClass == 0:
                        # right
                        mqttc.publish("testTopic1", "right   " + str(confidence))
                    elif predictedClass == 1:
                        # left
                        mqttc.publish("testTopic1", "left   " + str(confidence))
                    elif predictedClass == 2:
                        # forward
                        mqttc.publish("testTopic1", "forward   " + str(confidence))



                cv2.imshow("Omitted background", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        cv2.imshow("hand recognition  -  Author : Abbas Badiee", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True

def getPredictedClass():
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))

def showStatistics(predictedClass, confidence):
    global confidence_counter
    global flag

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "right"
    elif predictedClass == 1:
        className = "left"
    elif predictedClass == 2:
        className = "forward"

    cv2.putText(textImage, className,
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)



    if confidence_counter == 100:
        confidence_counter =0
    confidence_array[confidence_counter]= confidence*100
    confidence_counter+=1

    plt.plot(x_data, confidence_array)
    plt.xlabel('number of frames')
    plt.ylabel('Accuracy of ditection')
    sum =0
    for i in range(100):
        sum += confidence_array[i]
    accuracy_of_detection = sum/100
    if confidence_counter == 99:
        flag = True
    if flag == True:
        plt.title(str(accuracy_of_detection))
    #plt.legend()
    plt.show(block=False)
    plt.pause(0.05)
    plt.clf()

# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,3,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/GestureRecogModel.tfl")

main()
