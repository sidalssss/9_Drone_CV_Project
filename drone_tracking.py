import cv2
import numpy as np
import time

class DroneTracker:
    """Drone kameraları için nesne takip ve koordinat tahmin modülü."""
    def __init__(self, tracker_type="CSRT"):
        self.tracker = self._create_tracker(tracker_type)
        self.is_tracking = False
        self.history = []

    def _create_tracker(self, tracker_type):
        if tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        return cv2.TrackerKCF_create()

    def start_mission(self):
        """Takip görevini başlatır."""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if not ret:
            print("[HATA] Kamera akışı sağlanamadı.")
            return

        # Kullanıcıdan hedef seçmesini iste
        print("\nSidal AI Drone Mission: Lütfen takip edilecek nesneyi seçin (ROI)...")
        bbox = cv2.selectROI("Drone View - Select Target", frame, False)
        self.tracker.init(frame, bbox)
        self.is_tracking = True
        
        cv2.destroyWindow("Drone View - Select Target")

        while self.is_tracking:
            ret, frame = cap.read()
            if not ret: break

            success, box = self.tracker.update(frame)
            
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # Merkez koordinatı hesapla (PID Kontrolörü için girdi)
                center_x, center_y = x + w//2, y + h//2
                self.history.append((center_x, center_y))
                
                # Görselleştirme
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Telemetri Verileri (Simüle)
                cv2.putText(frame, f"Target Locked: ({center_x}, {center_y})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Stability: OK", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "TARGET LOST - RE-SCANNING", (100, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Sidal AI Drone Control Hub", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    drone_sys = DroneTracker()
    drone_sys.start_mission()
