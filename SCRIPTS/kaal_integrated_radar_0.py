import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 1. Load the model
model_path = r"D:\Github\Hackathons\KAAL\MODEL\yolov8m.pt"
logger.info("Loading YOLO model: %s", model_path)
model = YOLO(model_path)

def analyze_frequency(frame):
    # Convert image to grayscale (math doesn't care about color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Run the Fast Fourier Transform (The Black Magic)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Calculate the magnitude spectrum
    # Adding 1e-8 to prevent log(0) errors
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # We want to measure HIGH frequency noise (which adversarial patches use).
    # We create a mask to block out the center (low frequencies/normal shapes).
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask_size = 50 # Size of the center block
    
    # Extract only the outer frequencies (the invisible noise)
    high_freq_data = np.copy(magnitude_spectrum)
    high_freq_data[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    
    # Calculate the average noise level
    noise_score = np.mean(high_freq_data)
    
    return magnitude_spectrum, noise_score

def run_integrated_radar():
    cap = cv2.VideoCapture(0)
    logger.info("🚀 KAAL INTEGRATED RADAR (YOLO + FFT) ONLINE. PRESS 'q' TO QUIT.")
    
    # Threshold for attack (You will need to adjust this based on your camera/lighting)
    ATTACK_THRESHOLD = 110.0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize for faster math
        frame = cv2.resize(frame, (640, 480))
        
        # --- 1. YOLO INFERENCE ("The Eyes") ---
        # Run inference (conf=0.4 is a good balance for webcam, verbose=False stops console spam)
        results = model.predict(source=frame, conf=0.4, verbose=False)
        
        # Draw YOLO bounding boxes directly on the frame
        out_frame = results[0].plot()
        
        # --- 2. FFT INFERENCE ("The Secondary Brain") ---
        fft_vis, noise_score = analyze_frequency(frame)
        
        # Normalize the FFT visual so humans can see it
        fft_vis = cv2.normalize(fft_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        fft_vis_colored = cv2.applyColorMap(fft_vis, cv2.COLORMAP_JET)
        
        # --- THE UI DRAWING ---
        is_attack = noise_score > ATTACK_THRESHOLD
        
        color = (0, 0, 255) if is_attack else (0, 255, 0)
        status_text = "CRITICAL: ADVERSARIAL NOISE DETECTED" if is_attack else "SYSTEM NOMINAL: OPTICS CLEAR"
        
        # Draw on the YOLO output frame
        cv2.putText(out_frame, f"FFT NOISE SCORE: {noise_score:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out_frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if is_attack:
            cv2.rectangle(out_frame, (0, 0), (640, 480), (0, 0, 255), 10)
            
        # Show both the normal camera with YOLO and the "Predator Vision" FFT view
        cv2.imshow('KAAL - Tactical AI Vision (YOLOv8)', out_frame)
        cv2.imshow('KAAL - Frequency Domain Analysis (FFT)', fft_vis_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_integrated_radar()
