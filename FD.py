import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.9, maxHands=1)
fingerCount = 0
while True:
    # Get image frame
    success, img = cap.read()

    # Find the hand and its landmarks
    
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)
        #print("There are ", fingers1, "up")
       
        #print(type(fingers1))
        
        fingerCount = fingers1.count(1)

        print(fingerCount)
    
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if session.Process:
            volume.SetMasterVolume(((fingerCount*0.1)*2),None)
    
    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()