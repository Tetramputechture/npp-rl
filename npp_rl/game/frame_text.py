# Class that uses OpenCV to run OCR on the game screen and extract relevant text, given the
# current game state.

from typing import Optional
import numpy as np
from doctr.models import ocr_predictor, recognition_predictor
from doctr.io import Document, DocumentFile
import torch
import os

os.environ['USE_TORCH'] = 'YES'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True,
                      assume_straight_pages=True, preserve_aspect_ratio=True,
                      symmetric_pad=True, detect_orientation=False, straighten_pages=False,
                      detect_language=False, reco_bs=128).to(device)


def extract_text(frame: np.ndarray) -> Optional[str]:
    """Extract text from the given frame."""
    if frame is None:
        return None

    # Use the OCR model to extract text
    doc = model([frame])
    all_text = doc.render()

    # remove newlines
    all_text = all_text.replace('\n', ' ')

    # remove spaces between characters
    all_text = ''.join(all_text.split())

    # remove non-alphabetic characters
    all_text = ''.join(e for e in all_text if e.isalpha())

    return all_text


def all_frame_text(frame: np.ndarray):
    """Extract all text from the frame."""
    return extract_text(frame)


def main_menu_text(frame: np.ndarray):
    """Extract the main menu text from the frame."""
    # top left is (11, 65)
    # bottom right is (60, 102)
    return extract_text(frame[65:102, 11:60])
