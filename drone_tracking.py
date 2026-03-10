import cv2
import numpy as np
import logging
from typing import Tuple, Optional

# Log Yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DroneCV")

class DronePIDController:
    """
    Proportional-Integral-Derivative (PID) Kontrolörü.
    Drone'un nesne takibi sırasında stabil kalmasını ve pürüzsüz hareket etmesini sağlar.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, setpoint: float, measured_value: float, dt: float = 0.1) -> float:
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

class EnterpriseDroneTracker:
    """
    Gelişmiş Nesne Takip ve Stabilizasyon Sistemi.
    CSRT Tracker ve PID Kontrolörlerini birleştirerek otonom takip simülasyonu yapar.
    """
    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker = self._init_tracker(tracker_type)
        self.pid_x = DronePIDController(kp=0.45, ki=0.05, kd=0.1)
        self.pid_y = DronePIDController(kp=0.45, ki=0.05, kd=0.1)
        self.is_initialized = False

    def _init_tracker(self, t_type: str):
        if t_type == "CSRT":
            return cv2.TrackerCSRT_create()
        return cv2.TrackerKCF_create()

    def start_mission(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret: return

        # Target Selection (ROI)
        logger.info("Drone Misyonu: Takip edilecek hedefi seçin.")
        bbox = cv2.selectROI("Sidal-AI Drone View", frame, False)
        self.tracker.init(frame, bbox)
        self.is_initialized = True
        cv2.destroyWindow("Sidal-AI Drone View")

        while self.is_initialized:
            ret, frame = cap.read()
            if not ret: break

            success, box = self.tracker.update(frame)
            
            if success:
                (x, y, w, h) = [int(v) for v in box]
                center_x, center_y = x + w//2, y + h//2
                
                # PID Kontrolü (Görüntü merkezine sabitleme)
                frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
                correction_x = self.pid_x.compute(frame_center_x, center_x)
                correction_y = self.pid_y.compute(frame_center_y, center_y)
                
                # Görselleştirme (HUD)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), (255, 0, 0), 1)
                
                cv2.putText(frame, f"PID-X: {round(correction_x, 2)} PID-Y: {round(correction_y, 2)}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "TARGET LOST - SEARCHING...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Sidal AI - Drone Tracking HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EnterpriseDroneTracker()
    tracker.start_mission()
