import cv2

def drone_object_tracking():
    # Drone için nesne takip simülasyonu
    tracker = cv2.TrackerCSRT_create()
    cap = cv2.VideoCapture(0)
    
    print("Drone Takip Sistemi Aktif. Nesne seçmek için bir kare çizin.")
    
    ret, frame = cap.read()
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = cap.read()
        success, bbox = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Drone Tracking Target", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "LOST TARGET", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Drone CV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    drone_object_tracking()
