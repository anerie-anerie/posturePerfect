import cv2

cap = cv2.VideoCapture(1)

print('started')

while cap.isOpened():
    ret, frame = cap.read()
    print(frame)
    cv2.imshow('Mediapipe Feed', frame)
    print('opened')
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
