'''
Main program
@Author: David Vu

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import argparse
import sys
import json
import time
import numpy as np

TIMEOUT = 10 #10 seconds

'''
Description:
Images from Video Capture -> detect faces' regions -> crop those faces and align them 
    -> each cropped face is categorized in 3 types: Center, Left, Right 
    -> Extract 128D vectors( face features)
    -> Search for matching subjects in the dataset based on the types of face positions. 
    -> The preexisitng face 128D vector with the shortest distance to the 128D vector of the face on screen is most likely a match
    (Distance threshold is 0.6, percentage threshold is 70%)
    
'''

'''
facerec_128D.txt Data Structure:
{
"Person ID": {
    "Center": [[128D vector]],
    "Left": [[128D vector]],
    "Right": [[128D Vector]]
    }
}
This function basically does a simple linear search for 
^the 128D vector with the min distance to the 128D vector of the face on screen
'''
def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):
    '''
    :param features_arr: a list of 128d Features of all faces on screen
    :param positions: a list of face position types of all faces on screen
    :param thres: distance threshold
    :return: person name and percentage
    '''
    f = open('./facerec_128D.txt','r')
    data_set = json.loads(f.read())
    returnRes = []
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown"
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance
                    result = person
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))
    return returnRes    

if __name__ == '__main__':
    # aligner = AlignCustom()
    # extract_feature = FaceFeature(FaceRecGraph())
    face_detect = MTCNNDetect(FaceRecGraph(), scale_factor=2) #scale_factor, rescales image for faster detection

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 8)
    while True:
        ret, frame = cap.read()
        #u can certainly add a roi here but for the sake of a demo i'll just leave it as simple as this
        rects, landmarks = face_detect.detect_face(frame,40)#min face size is set to 80x80
        print(rects)
        if len(rects) > 0:
            for j in range(5):
                cv2.circle(frame, (int(landmarks[j,0]),int(int(landmarks[j+5,0]))), 4, (0,0,255), -1)
        # print(landmarks)
        # aligns = []
        # positions = []

        # for (i, rect) in enumerate(rects):
        #     aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
        #     cv2.imshow("aligned_face",aligned_face)
        #     if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
        #         aligns.append(aligned_face)
        #         positions.append(face_pos)
        #     else: 
        #         print("Align face failed") #log        
        # if(len(aligns) > 0):
        #     features_arr = extract_feature.get_features(aligns)
        #     recog_data = findPeople(features_arr,positions)
        #     for (i,rect) in enumerate(rects):
        #         cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
        #         cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
