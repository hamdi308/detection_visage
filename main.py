import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vid_cap=cv2.VideoCapture(0)
while True:
   successful_frame_read,frame=vid_cap.read()
   grau_scaled_vid=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   face_cordinated=trained_face_data.detectMultiScale(grau_scaled_vid)
   for(x,y,w,h) in face_cordinated:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),4)
     cv2.imshow('hamdi_face_detection', frame)
     key=cv2.waitKey(1)
     if key == 81 or key == 113 :
      vid_cap.release()   


"""
face_cordinated=trained_face_data.detectMultiScale(grau_scaled_img)
#print(face_cordinated)
for(x,y,w,h) in face_cordinated:
    cv2.rectangle(vid_cap,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),4)



print(face_cordinated)
cv2.imshow('hamdi_face_detection', vid_cap)
key=cv2.waitKey(1)
"""



















