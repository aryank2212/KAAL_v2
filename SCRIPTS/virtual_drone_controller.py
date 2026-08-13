import airsim
import time
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VirtualDroneController:
    """
    Connects to the AirSim simulation environment (Unreal Engine)
    and controls the virtual drone.
    """
    def __init__(self):
        logger.info("Connecting to Unreal Engine / AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            logger.info("Connected successfully!")
        except Exception as e:
            logger.error("Failed to connect to simulation: %s", e)
            logger.error("Please ensure Unreal Engine is running and you have pressed 'Play'.")
            sys.exit(1)

    def takeoff(self):
        logger.info("Taking off...")
        self.client.takeoffAsync().join()
        logger.info("Drone is airborne.")

    def hover(self):
        logger.info("Hovering...")
        self.client.hoverAsync().join()

    def get_camera_feed(self, camera_name="0", image_type=airsim.ImageType.Scene):
        """
        Retrieves the latest camera frame from the virtual drone.
        Returns the image as an OpenCV-compatible BGR numpy array.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, image_type, False, False)
        ])
        
        if len(responses) > 0 and responses[0].image_data_uint8:
            response = responses[0]
            # Convert binary image data to numpy 1D array
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
            # Reshape the 1D array to an OpenCV HxWxC 3D array (RGB/BGR)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
        return None

    def move_to_position(self, x, y, z, velocity):
        """
        Moves the drone to local coordinates (North/East/Down) synchronously.
        """
        logger.info("Moving to X:%s, Y:%s, Z:%s at %s m/s", x, y, z, velocity)
        self.client.moveToPositionAsync(x, y, z, velocity).join()

    def move_to_position_async(self, x, y, z, velocity):
        """
        Moves the drone asynchronously (doesn't block the camera feed).
        """
        logger.info("Patrolling towards X:%s, Y:%s, Z:%s at %s m/s", x, y, z, velocity)
        return self.client.moveToPositionAsync(x, y, z, velocity)

    def move_by_velocity_body_async(self, vx, vy, vz, duration):
        """
        Flies at a constant velocity RELATIVE to the drone's current heading.
        vx: forward/back, vy: right/left, vz: down/up
        """
        return self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration)
        
    def move_velocity_world_async(self, vx, vy, vz, yaw_rate, duration):
        """
        Calculates world velocity manually to bypass AirSim BodyFrame bugs.
        """
        import math
        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        pitch, roll, current_yaw = airsim.to_eularian_angles(q)
        
        vx_world = vx * math.cos(current_yaw) - vy * math.sin(current_yaw)
        vy_world = vx * math.sin(current_yaw) + vy * math.cos(current_yaw)
        
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        return self.client.moveByVelocityAsync(
            vx_world, vy_world, vz, duration, 
            airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode
        )

    def rotate_by_yaw_rate_async(self, yaw_rate, duration):
        """
        Spins the drone at a constant rate (degrees per second)
        """
        return self.client.rotateByYawRateAsync(yaw_rate, duration)

    def rotate_to_yaw(self, yaw_degrees):
        """
        Rotates the drone to an absolute yaw angle (degrees) synchronously.
        """
        logger.info("Rotating to yaw %s degrees", yaw_degrees)
        import math
        yaw_rad = math.radians(yaw_degrees)
        self.client.rotateToYawAsync(yaw_degrees).join()

    def land(self):
        logger.info("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

if __name__ == "__main__":
    # Test script: Connect, take off, spin around, land.
    drone = VirtualDroneController()
    drone.takeoff()
    
    # Move up 5 meters
    drone.move_to_position(0, 0, -5, 2)
    
    # Spin around
    for angle in range(0, 360, 45):
        drone.rotate_to_yaw(angle)
        time.sleep(1)
        
    drone.land()
    logger.info("Test complete.")
