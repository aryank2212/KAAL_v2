import os
import cv2
import logging
import numpy as np
import time
from ultralytics import YOLO
from pynput import keyboard
from virtual_drone_controller import VirtualDroneController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class SimulatedEye:
    def __init__(self, model_path):
        logger.info("Loading YOLO model from %s...", model_path)
        self.model = YOLO(model_path)
        self.drone = VirtualDroneController()
        
        # Setup global keyboard listener for smooth gaming controls
        self.keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.was_moving = False
        
    def on_press(self, key):
        try:
            self.keys.add(key.char.lower())
        except AttributeError:
            self.keys.add(key)
            
    def on_release(self, key):
        try:
            self.keys.discard(key.char.lower())
        except AttributeError:
            self.keys.discard(key)
        
    def detect_adversarial_patch_fft(self, image, bbox):
        """
        Simplified FFT Anomaly Detection.
        Crops the bounding box, converts to frequency domain, and checks for 
        unnatural high-frequency energy spikes typical of adversarial patches.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure box is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Perform FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Calculate energy in high frequencies
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        # Mask out the low frequencies (center)
        mask = np.ones((rows, cols), np.uint8)
        r = min(rows, cols) // 4
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        high_freq_energy = np.mean(magnitude_spectrum * mask)
        
        # Threshold for anomaly (this would normally be tuned)
        is_anomalous = high_freq_energy > 150.0 
        return is_anomalous

    def run_live_feed(self):
        logger.info("Initializing Virtual Drone...")
        self.drone = VirtualDroneController()
        
        # Take off
        self.drone.takeoff()
        
        logger.info("Ready for manual flight! Use W/A/S/D to move, R/F for Altitude, Space to Hover.")
        
        try:
            while True:
                # 1. Grab frame from virtual drone
                frame = self.drone.get_camera_feed()
                if frame is None:
                    logger.warning("Failed to grab frame from simulator.")
                    time.sleep(0.5)
                    continue
                
                # 2. Run YOLO Detection (Stream 1)
                results = self.model(frame, conf=0.15, verbose=False)
                
                # 3. Process Detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        name = self.model.names[cls]
                        
                        # Stream 2: FFT Anomaly Check
                        is_hacked = self.detect_adversarial_patch_fft(frame, b)
                        
                        # 4. Draw HUD
                        x1, y1, x2, y2 = map(int, b)
                        
                        if is_hacked:
                            # Flag as adversarial!
                            color = (0, 0, 255) # Red
                            label = f"WARNING: ADVERSARIAL PATCH DETECTED - {name}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # Normal detection
                            color = (0, 255, 0) # Green
                            label = f"{name} {conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 3. Collision Detection Check
                collision_info = self.drone.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    cv2.putText(frame, "CRASHED! PRESS BACKSPACE IN 3D WINDOW TO RESET!", (50, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    logger.warning("COLLISION DETECTED! Drone physics disabled until reset.")
                
                # Render HUD
                cv2.imshow("KAAL V2: Virtual Drone Feed", frame)
                
                # Exit and Manual Controls
                cv2.waitKey(1) # Keep OpenCV window refreshing
                
                if 'x' in self.keys:
                    break
                    
                vx, vy, vz, yaw = 0, 0, 0, 0
                speed = 5
                
                if 'w' in self.keys: vx = speed
                if 's' in self.keys: vx = -speed
                if 'a' in self.keys: vy = -speed
                if 'd' in self.keys: vy = speed
                
                if keyboard.Key.shift in self.keys or keyboard.Key.page_up in self.keys or keyboard.Key.up in self.keys: vz = -speed
                if keyboard.Key.ctrl_l in self.keys or keyboard.Key.page_down in self.keys or keyboard.Key.down in self.keys: vz = speed
                    
                if 'q' in self.keys or keyboard.Key.left in self.keys: yaw = -45
                if 'e' in self.keys or keyboard.Key.right in self.keys: yaw = 45
                
                is_moving = any([vx, vy, vz, yaw])
                current_input = (vx, vy, vz, yaw)
                
                if not hasattr(self, 'last_input'):
                    self.last_input = None
                
                if current_input != self.last_input:
                    self.last_input = current_input
                    
                    if is_moving:
                        self.was_moving = True
                        logger.debug("DEBUG INPUT: Sending Velocity -> X:%s Y:%s Z:%s Yaw:%s", vx, vy, vz, yaw)
                        self.drone.move_velocity_world_async(vx, vy, vz, yaw, 1000.0)
                    else:
                        if getattr(self, 'was_moving', False) or ' ' in self.keys:
                            logger.debug("DEBUG INPUT: Brakes applied.")
                            self.drone.move_velocity_world_async(0, 0, 0, 0, 1.0)
                            self.was_moving = False
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            logger.info("Landing and cleaning up...")
            cv2.destroyAllWindows()
            self.drone.land()

if __name__ == "__main__":
    # Ensure AirSim is running in Unreal Engine first!
    model_path = os.path.join(os.path.dirname(__file__), '..', 'MODEL', 'kaal_eye_v2.pt')
    eye = SimulatedEye(model_path)
    eye.run_live_feed()
