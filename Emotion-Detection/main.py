#importing
import cv2
from deepface import DeepFace
facecascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.XML')
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("can not open webcame")
while True:
    ret,frame=cap.read()
    ruselt=DeepFace.analyze(frame,actions=['emotion'], enforce_detection=False)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces=facecascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h)in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                ruselt['dominant_emotion'],
                (40,40),
                font,3,
                (0,255,0),
                2,
                cv2.LINE_4)
    cv2.imshow('Original video', frame)
    if cv2.waitKey(2) & 0xFF== ord('q'):
        break
cap.release()
cap.destroyAllWindows()
    
