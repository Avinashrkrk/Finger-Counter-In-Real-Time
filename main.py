import cv2
import time
import HandTracking as ht

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  
cap.set(4, hCam)  

detector = ht.HandDetector(detectionCon=0.8, maxHands=2)

tipIds = [4, 8, 12, 16, 20]

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture frame from camera.")
        break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmLists = []  
    for handNo in range(2): 
        lmList = detector.findPosition(img, handNo=handNo, draw=False)
        if lmList:
            lmLists.append(lmList)

    totalFingers = 0

    for lmList in lmLists:
        fingers = []

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: 
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers += fingers.count(1)

    cv2.rectangle(img, (20, 300), (200, 450), (0, 10, 0), cv2.FILLED)
    cv2.putText(
        img, str(totalFingers), (50, 418),
        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 10
    )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()