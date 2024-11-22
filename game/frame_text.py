# Class that uses OpenCV to run OCR on the game screen and extract relevant text, given the
# current game state.

import pytesseract
import numpy as np

from typing import Optional


def extract_text(frame: np.ndarray) -> Optional[str]:
    """Extract text from the given frame."""
    # Use pytesseract to extract text from the frame
    text = pytesseract.image_to_string(frame, config='--psm 11')

    # Remove non-alphanumeric characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())

    # Remove any leading or trailing whitespace
    text = text.strip()
    return text


class FrameText:
    def __init__(self):
        self.all_text = ''
        self.top_text = ''
        self.middle_text = ''
        self.bottom_text = ''

    def set_from_frame(self, frame: np.ndarray):
        self.all_text = extract_text(frame)
        height = frame.shape[0]
        third_height = height // 3
        self.top_text = extract_text(frame[:third_height, :])
        self.middle_text = extract_text(frame[third_height:2*third_height, :])
        self.bottom_text = extract_text(frame[2*third_height:, :])
