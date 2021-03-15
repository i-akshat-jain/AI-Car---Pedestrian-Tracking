import numpy as np
import cv2
#from pynput.keyboard import Key, Controller

 
#Our Image
img_file = 'Car image.png'
#video = cv2.VideoCapture('Tesla_Autopilot_Dashcam_Compilation_2018_Version.mp4')
video = cv2.VideoCapture('Pedestrians_Compilation.mp4')

#Our pre trained car classifier
#Create car tracker
car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('Pedestrian.xml')


#Run forever until care stops
while True:
    #Read the current frames
    (read_successful, frame) = video.read()

    #Save coding
    if read_successful:
        #Must convert to greysclae
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw the squares 
    for(x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w,y+h), (0 ,0,255), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0 ,255,), 2)

    #Draw the squares 
    for(x,y,w,h) in pedestrian:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    

    #Create the images with car spotted
    cv2.imshow("Car Detector", frame)

    #It does not auto close the slpit second of the image shown. It waits for the key to show
    key = cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release the video capture object
video.release()










    
"""#Create opencv image
img = cv2.imread(img_file)

#COnvert to grey scale (NEed for haar cascades)
black_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create a classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars
car = car_tracker.detectMultiScale(black_white)  

#Draw the squares 
for(x,y,w,h) in car:    
    cv2.rectangle(img, (x,y), (x+w,y+h), (0 ,0,255), 2)


#Create the images with car spotted
cv2.imshow("Car Detector", img)

#It does not auto close the slpit second of the image shown. It waits for the key to show
cv2.waitKey()

"""
print("Code Completed")