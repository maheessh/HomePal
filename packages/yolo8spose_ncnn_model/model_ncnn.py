import numpy as np
import ncnn
import cv2
import os

class YOLOv8PoseNCNNModel:
    """YOLOv8 Pose NCNN model wrapper for pose estimation"""
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(__file__)
        
        self.model_dir = model_dir
        self.net = None
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        self._load_model()
        
        # COCO pose keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def _load_model(self):
        """Load the NCNN model"""
        try:
            self.net = ncnn.Net()
            param_path = os.path.join(self.model_dir, "model.ncnn.param")
            bin_path = os.path.join(self.model_dir, "model.ncnn.bin")
            
            if not os.path.exists(param_path) or not os.path.exists(bin_path):
                raise FileNotFoundError(f"Model files not found in {self.model_dir}")
            
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            print(f"✅ YOLOv8 Pose NCNN model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 Pose NCNN model: {e}")
            self.net = None
    
    def predict(self, image):
        """
        Predict poses in input image
        Args:
            image: Input image (BGR format)
        Returns:
            poses: List of pose detections with bbox and keypoints
        """
        if self.net is None:
            return []
        
        try:
            h, w = image.shape[:2]
            
            # Preprocess image
            resized = cv2.resize(image, self.input_size)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Create input tensor
            input_mat = ncnn.Mat.from_pixels(normalized, ncnn.Mat.PixelType.PIXEL_RGB,
                                           self.input_size[0], self.input_size[1])
            
            poses = []
            
            # Run inference
            with self.net.create_extractor() as ex:
                ex.input("in0", input_mat)
                _, out0 = ex.extract("out0")
                
                # Convert output to numpy array
                output = np.array(out0)
                
                if output.size > 0:
                    # Process detections (assuming format: [batch, num_detections, features])
                    # Features: 4 (bbox) + 1 (conf) + 17*3 (keypoints)
                    if len(output.shape) == 2:
                        num_detections = output.shape[0]
                        for i in range(num_detections):
                            detection = output[i]
                            
                            if len(detection) < 56:  # 4 + 1 + 51
                                continue
                            
                            # Extract bbox and confidence
                            x_center, y_center, width, height = detection[:4]
                            confidence = detection[4]
                            
                            if confidence < self.confidence_threshold:
                                continue
                            
                            # Scale coordinates back to original frame
                            x_center *= w / self.input_size[0]
                            y_center *= h / self.input_size[1]
                            width *= w / self.input_size[0]
                            height *= h / self.input_size[1]
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            
                            # Extract keypoints
                            keypoints = []
                            for j in range(17):
                                kp_idx = 5 + j * 3
                                if kp_idx + 2 < len(detection):
                                    kp_x = detection[kp_idx] * w / self.input_size[0]
                                    kp_y = detection[kp_idx + 1] * h / self.input_size[1]
                                    kp_vis = detection[kp_idx + 2]
                                    keypoints.append([kp_x, kp_y, kp_vis])
                                else:
                                    keypoints.append([0, 0, 0])
                            
                            poses.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'keypoints': keypoints
                            })
            
            return poses
                
        except Exception as e:
            print(f"❌ Pose inference error: {e}")
            return []
    
    def draw_poses(self, image, poses):
        """Draw poses on image"""
        result_image = image.copy()
        
        for pose in poses:
            bbox = pose['bbox']
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw confidence
            cv2.putText(result_image, f"{confidence:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw keypoints
            for i, kp in enumerate(keypoints):
                if kp[2] > 0.5:  # Only draw visible keypoints
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(result_image, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(result_image, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return result_image

def test_inference():
    """Test the model with a sample image"""
    model = YOLOv8PoseNCNNModel()
    
    # Create a test image (random noise)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    poses = model.predict(test_image)
    print(f"Detected {len(poses)} poses")
    
    for i, pose in enumerate(poses):
        print(f"Pose {i}: confidence={pose['confidence']:.3f}, "
              f"bbox={pose['bbox']}")
    
    return poses

if __name__ == "__main__":
    test_inference()
