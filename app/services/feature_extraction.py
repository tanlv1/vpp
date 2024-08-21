import cv2
import numpy as np
from typing import List


def extract_frames(video_path: str, frame_rate: int = 1) -> List[np.ndarray]:
    """
    Extract frames from a video at a specified frame rate.

    :param video_path: Path to the video file.
    :param frame_rate: Number of frames to extract per second.
    :return: List of extracted frames as numpy arrays.
    """
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frames.append(frame)

        count += 1

    video_capture.release()
    return frames


def extract_features(frames: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from a list of frames using a pre-trained deep learning model.

    :param frames: List of frames as numpy arrays.
    :return: Extracted feature vectors.
    """
    # This is a placeholder for actual feature extraction logic.
    # You would typically use a pre-trained model like ResNet, VGG, etc.
    feature_vectors = []

    for frame in frames:
        # Example: Resize frame, normalize, and pass it through a model.
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        # Assuming `model` is a pre-loaded deep learning model.
        # feature_vector = model.predict(normalized_frame[np.newaxis, ...])
        # Placeholder feature vector
        feature_vector = np.mean(normalized_frame, axis=(0, 1))
        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)
