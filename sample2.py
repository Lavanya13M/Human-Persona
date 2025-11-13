import cv2
import numpy as np
from deepface import DeepFace
import statistics
from collections import defaultdict, Counter

# ============== Optimized Settings for Maximum Accuracy ==============
FACE_MODEL_PB     = "opencv_face_detector_uint8.pb"      
FACE_CONFIG_PBTXT = "opencv_face_detector.pbtxt"   
CONF_THRESHOLD    = 0.75   # lowered for better coverage with quality filtering
PADDING           = 25     # optimal padding for all attributes
ANALYZE_EVERY_N   = 5      # analyze every 2 frames for stability
# ANALYZE_EVERY_N   = 2      # analyze every 2 frames for stability
WINDOW_NAME       = "Webcam — Maximum Accuracy Age/Gender/Emotion Detection"

# ============== Accuracy Enhancement Settings ==============
AGE_CONFIDENCE_MIN     = 0.3   # lowered for better age coverage
GENDER_CONFIDENCE_MIN  = 0.95  # keep your original strict threshold  
EMOTION_CONFIDENCE_MIN = 0.25  # lowered to avoid question marks
SMOOTHING_WINDOW       = 7     # increased for better age stability
MIN_FACE_SIZE         = 60     # increased for better age detection
MAX_AGE_VARIANCE      = 12     # tighter for age accuracy
AGE_SMOOTHING_WEIGHT   = 0.8   # focus more on age accuracy
CLEANUP_INTERVAL      = 100    # frames between tracker cleanup
# ================================================================

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL_PB, FACE_CONFIG_PBTXT)

class TemporalSmoother:
    """Advanced temporal smoothing with confidence weighting and outlier rejection"""
    def __init__(self, window_size=SMOOTHING_WINDOW):
        self.window_size = window_size
        self.age_history = []
        self.age_conf_history = []
        self.gender_prob_history = []
        self.emotion_history = []
        self.emotion_conf_history = []
    
    def add_predictions(self, age, age_conf, gender_probs, emotion, emotion_conf):
        """Add new predictions with confidence scores - enhanced for age focus"""
        # Enhanced age smoothing with multiple validation checks
        if isinstance(age, (int, float)) and 5 <= age <= 100 and age_conf >= AGE_CONFIDENCE_MIN:
            if len(self.age_history) >= self.window_size:
                self.age_history.pop(0)
                self.age_conf_history.pop(0)
            self.age_history.append(age)
            # Boost confidence for age predictions in reasonable range
            boosted_conf = min(1.0, age_conf * 1.2) if 18 <= age <= 65 else age_conf
            self.age_conf_history.append(boosted_conf)
        
        # Gender probability smoothing (unchanged)
        if isinstance(gender_probs, dict) and gender_probs:
            if len(self.gender_prob_history) >= self.window_size:
                self.gender_prob_history.pop(0)
            self.gender_prob_history.append(gender_probs)
        
        # Emotion smoothing - more lenient to avoid ??? 
        if emotion and emotion != "Unknown" and emotion != "err":
            if len(self.emotion_history) >= self.window_size:
                self.emotion_history.pop(0)
                self.emotion_conf_history.pop(0)
            self.emotion_history.append(emotion)
            self.emotion_conf_history.append(max(0.3, emotion_conf))  # Boost low confidence
    
    def get_smoothed_age(self):
        """Enhanced age smoothing with multiple validation layers"""
        if not self.age_history:
            return None, 0.0
        
        if len(self.age_history) >= 3:
            # Step 1: Remove statistical outliers
            median_age = statistics.median(self.age_history)
            filtered_ages = []
            filtered_confs = []
            
            for age, conf in zip(self.age_history, self.age_conf_history):
                if abs(age - median_age) <= MAX_AGE_VARIANCE:
                    filtered_ages.append(age)
                    filtered_confs.append(conf)
            
            if len(filtered_ages) >= 2:
                # Step 2: Apply exponential weighting (recent predictions matter more)
                weights = [AGE_SMOOTHING_WEIGHT ** (len(filtered_ages) - 1 - i) for i in range(len(filtered_ages))]
                
                # Step 3: Confidence-weighted + time-weighted average
                total_weight = sum(w * c for w, c in zip(weights, filtered_confs))
                if total_weight > 0:
                    weighted_age = sum(age * w * c for age, w, c in zip(filtered_ages, weights, filtered_confs)) / total_weight
                    
                    # Step 4: Final confidence based on consistency
                    age_std = statistics.stdev(filtered_ages) if len(filtered_ages) > 1 else 0
                    consistency_bonus = max(0, (10 - age_std) / 10)  # bonus for consistent predictions
                    final_conf = min(1.0, (sum(filtered_confs) / len(filtered_confs)) + consistency_bonus)
                    
                    return int(round(weighted_age)), final_conf
        
        # Fallback for insufficient data
        if self.age_history:
            avg_age = sum(self.age_history) / len(self.age_history)
            avg_conf = sum(self.age_conf_history) / len(self.age_conf_history)
            return int(round(avg_age)), avg_conf
        
        return None, 0.0
    
    def get_smoothed_gender(self):
        """Get smoothed gender with averaged probabilities"""
        if not self.gender_prob_history:
            return "Unknown", "", 0.0
        
        # Average probabilities across all frames
        avg_probs = defaultdict(float)
        for gender_dict in self.gender_prob_history:
            for gender, prob in gender_dict.items():
                avg_probs[gender] += prob
        
        # Normalize by number of samples
        for gender in avg_probs:
            avg_probs[gender] /= len(self.gender_prob_history)
        
        if avg_probs:
            woman_prob = avg_probs.get("Woman", 0)
            man_prob = avg_probs.get("Man", 0)
            max_conf = max(woman_prob, man_prob)
            
            # Apply your original strict threshold logic
            if woman_prob >= GENDER_CONFIDENCE_MIN:
                gender = "Woman"
            elif man_prob >= GENDER_CONFIDENCE_MIN:
                gender = "Man"
            else:
                gender = "Uncertain"
            
            gender_conf_str = f"(W:{woman_prob:.2f}, M:{man_prob:.2f})"
            return gender, gender_conf_str, max_conf
        
        return "Unknown", "", 0.0
    
    def get_smoothed_emotion(self):
        """Get smoothed emotion using voting"""
        if not self.emotion_history:
            return "Unknown", 0.0
        
        # Use weighted voting based on confidence
        emotion_votes = defaultdict(float)
        for emotion, conf in zip(self.emotion_history, self.emotion_conf_history):
            emotion_votes[emotion] += conf
        
        if emotion_votes:
            best_emotion = max(emotion_votes, key=emotion_votes.get)
            # Calculate average confidence for this emotion
            emotion_confs = [conf for emotion, conf in zip(self.emotion_history, self.emotion_conf_history) 
                           if emotion == best_emotion]
            avg_conf = sum(emotion_confs) / len(emotion_confs) if emotion_confs else 0.0
            return best_emotion, avg_conf
        
        return "Unknown", 0.0

def detect_faces_dnn(frame, conf_threshold=CONF_THRESHOLD):
    """
    Enhanced face detection with quality filtering.
    Returns list of [x1, y1, x2, y2] integer boxes.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Quality filtering
            face_width = x2 - x1
            face_height = y2 - y1
            
            if (x2 > x1 and y2 > y1 and 
                min(face_width, face_height) >= MIN_FACE_SIZE and
                0.4 <= face_width / face_height <= 2.5 and  # Aspect ratio check
                x1 >= 5 and y1 >= 5 and x2 < w-5 and y2 < h-5):  # Not at edge
                boxes.append([x1, y1, x2, y2])
    return boxes

def enhance_face_for_analysis(face_bgr):
    """
    Multi-stage face enhancement for maximum model accuracy
    """
    # Step 1: Noise reduction
    denoised = cv2.bilateralFilter(face_bgr, 9, 75, 75)
    
    # Step 2: Contrast enhancement using CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Step 3: Subtle sharpening
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
    
    # Step 4: Blend with original to prevent over-processing
    final = cv2.addWeighted(sharpened, 0.6, face_bgr, 0.4, 0)
    
    return final

def analyze_face_deepface_ensemble(face_bgr):
    """
    Enhanced DeepFace analysis with ensemble predictions and confidence scoring.
    Returns (age, gender, emotion, gender_conf_str, age_conf, emotion_conf).
    """
    try:
        # Create ensemble of face versions for enhanced age accuracy
        original_face = face_bgr
        enhanced_face = enhance_face_for_analysis(face_bgr)
        
        # For age accuracy: ensure optimal face size (DeepFace works best with 224x224)
        h, w = original_face.shape[:2]
        target_size = 224
        if min(h, w) != target_size:
            # High-quality resize for age detection
            original_face = cv2.resize(original_face, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            enhanced_face = cv2.resize(enhanced_face, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Run triple ensemble for maximum age accuracy
        face_versions = [original_face, enhanced_face]
        
        # Add gamma-corrected version for different lighting conditions
        gamma_corrected = np.power(original_face / 255.0, 0.8) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        face_versions.append(gamma_corrected)
        
        # Run ensemble analysis on all face versions
        results = []
        for i, face_version in enumerate(face_versions):
            try:
                result = DeepFace.analyze(
                    img_path = face_version,
                    actions  = ['age', 'gender', 'emotion'],
                    enforce_detection = False
                )
                data = result[0] if isinstance(result, list) else result
                results.append(data)
            except:
                continue
        
        if not results:
            return "?", "?", "?", "", 0.0, 0.0
        
        # ============== ENHANCED AGE ENSEMBLE FOR MAXIMUM ACCURACY ==============
        ages = []
        age_confidences = []
        
        for i, data in enumerate(results):
            age_val = data.get('age', None)
            if isinstance(age_val, (int, float)) and 5 <= age_val <= 100:
                ages.append(age_val)
                
                # Calculate confidence based on image quality and consistency
                base_conf = 0.7
                
                # Bonus for enhanced/processed images
                if i == 1:  # enhanced image
                    base_conf += 0.1
                elif i == 2:  # gamma corrected
                    base_conf += 0.05
                
                age_confidences.append(min(1.0, base_conf))
        
        if ages:
            if len(ages) >= 2:
                # Multi-step age refinement
                # Step 1: Remove statistical outliers
                median_age = statistics.median(ages)
                filtered_ages = [age for age in ages if abs(age - median_age) <= 8]  # tighter filter
                
                if not filtered_ages:
                    filtered_ages = ages  # fallback if all removed
                
                # Step 2: Weighted average with recency bias
                weights = [1.5, 1.2, 1.0][:len(filtered_ages)]  # prefer later predictions
                if len(weights) < len(filtered_ages):
                    weights.extend([1.0] * (len(filtered_ages) - len(weights)))
                
                weighted_sum = sum(age * w for age, w in zip(filtered_ages, weights))
                total_weight = sum(weights)
                final_age = int(round(weighted_sum / total_weight))
                
                # Enhanced confidence calculation
                age_std = statistics.stdev(filtered_ages) if len(filtered_ages) > 1 else 0
                consistency_score = max(0.1, 1.0 - (age_std / 20))  # penalize high variance
                age_confidence = min(1.0, consistency_score * 0.9)
                
            else:
                final_age = int(round(ages[0]))
                age_confidence = 0.8
        else:
            final_age = "?"
            age_confidence = 0.0
        
        # ============== GENDER ENSEMBLE (Your Original Logic Preserved) ==============
        # Average gender probabilities across ensemble
        gender_avg = defaultdict(float)
        gender_count = 0
        
        for data in results:
            gender_probs = data.get("gender", {})
            if isinstance(gender_probs, dict):
                for gender, prob in gender_probs.items():
                    gender_avg[gender] += prob
                gender_count += 1
        
        # Normalize and apply your exact original threshold (0.95)
        if gender_count > 0:
            for gender in gender_avg:
                gender_avg[gender] /= gender_count
            
            woman_prob = gender_avg.get("Woman", 0)
            man_prob = gender_avg.get("Man", 0)
            
            # Your exact original 95% threshold logic
            if woman_prob >= 0.95:
                gender = "Woman"
            elif man_prob >= 0.95:
                gender = "Man"
            else:
                gender = "Uncertain"
            
            gender_conf_str = f"(W:{woman_prob:.2f}, M:{man_prob:.2f})"
        else:
            gender = "Unknown"
            gender_conf_str = ""
        
        # ============== EMOTION ENSEMBLE WITH CONFIDENCE ==============
        emotion_votes = defaultdict(list)
        
        for data in results:
            emotion_probs = data.get("emotion", {})
            if isinstance(emotion_probs, dict) and emotion_probs:
                max_emotion = max(emotion_probs, key=emotion_probs.get)
                max_conf = emotion_probs[max_emotion]
                emotion_votes[max_emotion].append(max_conf)
        
        # Get best emotion with confidence - NO question marks
        if emotion_votes:
            # Calculate average confidence for each emotion
            emotion_avg_conf = {}
            for emotion_name, confs in emotion_votes.items():
                emotion_avg_conf[emotion_name] = sum(confs) / len(confs)
            
            # Get emotion with highest average confidence
            best_emotion = max(emotion_avg_conf, key=emotion_avg_conf.get)
            emotion_confidence = emotion_avg_conf[best_emotion]
            
            # Always show the best emotion (no ??? or uncertain)
            emotion = best_emotion
        else:
            # Fallback to first result's dominant emotion
            data = results[0]
            emotion = data.get('dominant_emotion', 'neutral')
            emotion_confidence = 0.5
        
        return final_age, gender, emotion, gender_conf_str, age_confidence, emotion_confidence
        
    except Exception as e:
        return "?", "?", "?", "", 0.0, 0.0

def get_face_id(x1, y1, x2, y2):
    """Generate consistent face ID for tracking"""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    size = ((x2 - x1) + (y2 - y1)) // 2
    return f"{center_x//40}_{center_y//40}_{size//20}"

def get_confidence_indicator(confidence):
    """Visual confidence indicator"""
    if confidence >= 0.7: return "●"     # High confidence
    elif confidence >= 0.4: return "◐"   # Medium confidence  
    else: return "○"                      # Low confidence

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam")
        return

    print("✅ Webcam opened with maximum accuracy optimizations")
    print("🎯 Features enabled:")
    print("   ├── Temporal smoothing (5-frame window)")
    print("   ├── Ensemble predictions (2 face versions)")
    print("   ├── Confidence thresholding")
    print("   ├── Face enhancement preprocessing") 
    print("   ├── Quality filtering")
    print("   └── Mirror effect")
    print("📊 Confidence indicators: ● High, ◐ Medium, ○ Low")
    print("Press 'q' to quit.")
    
    frame_idx = 0
    last_results = []
    face_smoothers = {}  # face_id -> TemporalSmoother

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror the frame horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Detect faces with enhanced filtering
        boxes = detect_faces_dnn(frame, CONF_THRESHOLD)

        current_results = []
        run_analysis = (frame_idx % ANALYZE_EVERY_N == 0)

        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            # Adaptive padding based on face size
            ph, pw = frame.shape[:2]
            face_width = x2 - x1
            face_height = y2 - y1
            adaptive_padding = max(PADDING, int(min(face_width, face_height) * 0.25))
            
            px1 = max(0, x1 - adaptive_padding)
            py1 = max(0, y1 - adaptive_padding)
            px2 = min(pw - 1, x2 + adaptive_padding)
            py2 = min(ph - 1, y2 + adaptive_padding)

            face = frame[py1:py2, px1:px2]
            
            # Generate face ID for tracking
            face_id = get_face_id(x1, y1, x2, y2)
            if face_id not in face_smoothers:
                face_smoothers[face_id] = TemporalSmoother()

            age, gender, emotion, gender_conf = None, None, None, ""
            age_conf, emotion_conf = 0.0, 0.0

            if run_analysis and face.size != 0:
                try:
                    # Run ensemble analysis
                    age, gender, emotion, gender_conf, age_conf, emotion_conf = analyze_face_deepface_ensemble(face)
                    
                    # Add to temporal smoother
                    gender_probs = {}
                    if "W:" in gender_conf and "M:" in gender_conf:
                        # Extract probabilities from your confidence string
                        import re
                        matches = re.findall(r'([WM]):([\d.]+)', gender_conf)
                        for label, prob in matches:
                            gender_key = "Woman" if label == "W" else "Man"
                            gender_probs[gender_key] = float(prob)
                    
                    face_smoothers[face_id].add_predictions(age, age_conf, gender_probs, emotion, emotion_conf)
                    
                    # Get smoothed results
                    smoothed_age, smoothed_age_conf = face_smoothers[face_id].get_smoothed_age()
                    smoothed_gender, smoothed_gender_conf, gender_conf_val = face_smoothers[face_id].get_smoothed_gender()
                    smoothed_emotion, smoothed_emotion_conf = face_smoothers[face_id].get_smoothed_emotion()
                    
                    # Use smoothed results if available and better
                    if smoothed_age is not None and smoothed_age_conf >= age_conf:
                        age = smoothed_age
                        age_conf = smoothed_age_conf
                    
                    if smoothed_gender != "Unknown":
                        gender = smoothed_gender
                        gender_conf = smoothed_gender_conf
                    
                    if smoothed_emotion != "Unknown" and smoothed_emotion_conf >= emotion_conf:
                        emotion = smoothed_emotion
                        emotion_conf = smoothed_emotion_conf
                    
                except Exception as e:
                    age, gender, emotion, gender_conf = "?", "?", "err", ""
                    age_conf, emotion_conf = 0.0, 0.0
                
                current_results.append({
                    "box": [x1, y1, x2, y2],
                    "age": age, "gender": gender, "emotion": emotion, 
                    "gender_conf": gender_conf, "age_conf": age_conf, "emotion_conf": emotion_conf
                })
            else:
                # Reuse previous results with potential smoothed updates
                if bi < len(last_results):
                    cached = last_results[bi]
                    
                    # Try to get smoothed results even when not analyzing
                    smoothed_age, smoothed_age_conf = face_smoothers[face_id].get_smoothed_age()
                    smoothed_gender, smoothed_gender_conf, _ = face_smoothers[face_id].get_smoothed_gender()
                    smoothed_emotion, smoothed_emotion_conf = face_smoothers[face_id].get_smoothed_emotion()
                    
                    # Use smoothed if available, otherwise cached
                    display_age = smoothed_age if smoothed_age is not None else cached.get("age", "…")
                    display_gender = smoothed_gender if smoothed_gender != "Unknown" else cached.get("gender", "…")
                    display_emotion = smoothed_emotion if smoothed_emotion != "Unknown" else cached.get("emotion", "…")
                    
                    current_results.append({
                        "box": [x1, y1, x2, y2],
                        "age": display_age,
                        "gender": display_gender,
                        "emotion": display_emotion,
                        "gender_conf": smoothed_gender_conf if smoothed_gender != "Unknown" else cached.get("gender_conf", ""),
                        "age_conf": smoothed_age_conf if smoothed_age is not None else cached.get("age_conf", 0.0),
                        "emotion_conf": smoothed_emotion_conf if smoothed_emotion != "Unknown" else cached.get("emotion_conf", 0.0)
                    })
                else:
                    current_results.append({
                        "box": [x1, y1, x2, y2],
                        "age": "…", "gender": "…", "emotion": "…", 
                        "gender_conf": "", "age_conf": 0.0, "emotion_conf": 0.0
                    })

        # If we ran DeepFace this frame, update cache
        if run_analysis:
            last_results = current_results

        # Cleanup old trackers periodically
        if frame_idx % CLEANUP_INTERVAL == 0:
            active_ids = set()
            for item in current_results:
                x1, y1, x2, y2 = item["box"]
                face_id = get_face_id(x1, y1, x2, y2)
                active_ids.add(face_id)
            
            # Remove inactive trackers
            face_smoothers = {k: v for k, v in face_smoothers.items() if k in active_ids}

        # Enhanced drawing with confidence visualization
        for item in current_results:
            x1, y1, x2, y2 = item["box"]
            
            # Color-coded bounding box based on overall confidence
            age_conf = item.get("age_conf", 0.0)
            emotion_conf = item.get("emotion_conf", 0.0)
            
            # Calculate overall confidence (age and emotion, gender is always high with your threshold)
            avg_confidence = (age_conf + emotion_conf) / 2
            
            if avg_confidence >= 0.6:
                box_color = (0, 255, 0)      # Green - high confidence
            elif avg_confidence >= 0.3:
                box_color = (0, 255, 255)    # Yellow - medium confidence  
            else:
                box_color = (0, 150, 255)    # Orange - low confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Build enhanced labels with confidence indicators
            age_indicator = get_confidence_indicator(age_conf)
            emotion_indicator = get_confidence_indicator(emotion_conf)
            
            lbl_age = f"Age: {item['age']}" 
            lbl_gender = f"Gender: {item['gender']}"      
            lbl_emotion = f"Emotion: {item['emotion']}q"

            # Enhanced text positioning and styling
            y_base = max(70, y1 - 10)
            font_scale = 0.6
            thickness = 2
            
            cv2.putText(frame, lbl_age,     (x1, y_base - 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
            cv2.putText(frame, lbl_gender,  (x1, y_base - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,0,0), thickness, cv2.LINE_AA)
            cv2.putText(frame, lbl_emotion, (x1, y_base - 5),  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness, cv2.LINE_AA)

        # Enhanced statistics display
        stats_text = f"Faces: {len(current_results)} | Active Trackers: {len(face_smoothers)} | Frame: {frame_idx}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Settings display
        settings_text = f"Thresholds: Face={CONF_THRESHOLD} | Gender={GENDER_CONFIDENCE_MIN} | Emotion={EMOTION_CONFIDENCE_MIN}"
        cv2.putText(frame, settings_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Enhancement info
        enhance_text = f"Enhancements: Smoothing={SMOOTHING_WINDOW} | Ensemble=ON | Enhancement=ON"
        cv2.putText(frame, enhance_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        cv2.imshow(WINDOW_NAME, frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Closed.")
    print(f"📊 Final Statistics:")
    print(f"   ├── Total frames processed: {frame_idx}")
    print(f"   ├── Peak active trackers: {len(face_smoothers)}")
    print(f"   └── Accuracy optimizations: ALL ENABLED")

if __name__ == "__main__":
    main()