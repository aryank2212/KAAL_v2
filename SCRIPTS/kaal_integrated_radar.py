import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 1. Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MODEL', 'yolov8m.pt')
logger.info("Loading YOLO model: %s", model_path)
model = YOLO(model_path)

# --- DISTANCE ESTIMATION CONSTANTS ---
# Known widths of objects in centimeters (approximate)
KNOWN_WIDTHS = {
    "person": 50,
    "cell phone": 7,
    "laptop": 35,
    "bottle": 8,
    "cup": 8,
    "chair": 50,
    "backpack": 30,
    "car": 180,
}
# Assumed focal length (needs calibration for better accuracy, but this is a starting point)
# Formula: focal_length = (pixel_width * real_distance) / real_width
# Assuming at 100cm, a person (50cm) takes up ~300 pixels on a 640px wide frame
FOCAL_LENGTH = 600 

def estimate_distance(pixel_width, real_width):
    if pixel_width == 0: return 0
    return (real_width * FOCAL_LENGTH) / pixel_width

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

def run_integrated_radar(output_path="kaal_output_video.mp4"):
    cap = cv2.VideoCapture(0)
    logger.info("🚀 KAAL INTEGRATED RADAR (YOLO + FFT) ONLINE. PRESS 'q' TO QUIT.")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    
    # Threshold for attack (You will need to adjust this based on your camera/lighting)
    ATTACK_THRESHOLD = 110.0 
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp = time.strftime("%H:%M:%S")
        
        # Resize for faster math
        frame = cv2.resize(frame, (640, 480))
        
        # --- 1. YOLO INFERENCE ("The Eyes") ---
        results = model.predict(source=frame, conf=0.4, verbose=False)
        out_frame = results[0].plot()
        
        detections_report = []
        
        # Distance Estimation Logic
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pixel_width = x2 - x1
                
                real_width = KNOWN_WIDTHS.get(cls_name.lower(), 50)
                distance_cm = estimate_distance(pixel_width, real_width)
                distance_m = distance_cm / 100.0
                
                detections_report.append(f"{cls_name.upper()} at {distance_m:.2f}m")
                
                label = f"Dist: {distance_m:.2f}m"
                cv2.putText(out_frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # --- 2. FFT INFERENCE ("The Secondary Brain") ---
        fft_vis, noise_score = analyze_frequency(frame)
        
        # Normalize the FFT visual so humans can see it
        fft_vis = cv2.normalize(fft_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        fft_vis_colored = cv2.applyColorMap(fft_vis, cv2.COLORMAP_JET)
        
        # --- THE UI DRAWING ---
        is_attack = noise_score > ATTACK_THRESHOLD
        status_text = "CRITICAL: ADVERSARIAL NOISE DETECTED" if is_attack else "SYSTEM NOMINAL: OPTICS CLEAR"
        color = (0, 0, 255) if is_attack else (0, 255, 0)
        
        # --- REAL-TIME TERMINAL REPORT ---
        logger.info("\n[FRAME %04d | %s]", frame_count, timestamp)
        logger.info("📡 STATUS: %s", status_text)
        logger.info("📊 NOISE SCORE: %.2f", noise_score)
        
        if detections_report:
            logger.info("🎯 TARGETS DETECTED (%d):", len(detections_report))
            for det in detections_report:
                logger.info("   - %s", det)
        else:
            logger.info("🔭 TARGETS: NONE")
        logger.info("-" * 40)

        # Draw on the YOLO output frame
        cv2.putText(out_frame, f"FFT NOISE SCORE: {noise_score:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out_frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if is_attack:
            cv2.rectangle(out_frame, (0, 0), (640, 480), (0, 0, 255), 10)
            
        # Create a side-by-side frame for saving
        # Resize fft_vis_colored to match out_frame height if they differ
        fft_vis_resized = cv2.resize(fft_vis_colored, (out_frame.shape[1], out_frame.shape[0]))
        combined_output = np.hstack((out_frame, fft_vis_resized))

        # Initialize VideoWriter after the first frame size is known
        if out_video is None:
            height, width, _ = combined_output.shape
            out_video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            logger.info("Recording to: %s", output_path)

        # Save the combined frame to the video file
        out_video.write(combined_output)
            
        # Show both the normal camera with YOLO and the "Predator Vision" FFT view
        cv2.imshow('KAAL - Tactical AI Vision (YOLOv8)', out_frame)
        cv2.imshow('KAAL - Frequency Domain Analysis (FFT)', fft_vis_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_integrated_radar()
