# Class that uses OpenCV to run OCR on the game screen and extract relevant text, given the
# current game state.

import pytesseract
import numpy as np

from typing import Optional


def extract_text(frame: np.ndarray) -> Optional[str]:
    """Extract text from the given frame."""
    # if frame is None:
    #     return None

    # # Use pytesseract to extract text from the frame
    # text = pytesseract.image_to_string(frame, config='--psm 11')

    # # Remove non-alphanumeric characters
    # text = ''.join(e for e in text if e.isalnum() or e.isspace())

    # # Replace multiple spaces with a single space
    # text = ' '.join(text.split())

    # # Remove any leading or trailing whitespace
    # text = text.strip()
    # return text
    return ''


def all_frame_text(frame: np.ndarray):
    """Extract all text from the frame."""
    return extract_text(frame)


def level_playing_center_text(frame: np.ndarray):
    """Extract the level center title from the frame."""
    # Our coordinates are the rectangle where the top left coordinate is (178, 273)
    # and the bottom right coordinate is (470, 302)
    # We extract the text from this rectangle
    level_center_title = extract_text(frame[280:302, 170:470])
    return level_center_title


def main_menu_text(frame: np.ndarray):
    """Extract the main menu text from the frame."""
    # top left is (11, 65)
    # bottom right is (60, 102)
    return extract_text(frame[65:102, 11:60])
