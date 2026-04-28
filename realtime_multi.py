import cv2
import numpy as np
import torch
import sounddevice as sd
import torchaudio
from torchvision import transforms
from PIL import Image
from model_pt import FERNet
from utils_img_pt import get_dataloaders
from mfcc_mlp import MFCCMLP
from fuse import fuse_probs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AUDIO_SR = 16000
AUDIO_DUR = 4.0
AUDIO_LEN = int(AUDIO_SR * AUDIO_DUR)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_model = FERNet().to(DEVICE)
face_model.load_state_dict(torch.load('fer_net_best.pt', map_location=DEVICE))
face_model.eval()

face_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

audio_model = MFCCMLP(n_mfcc=40, hidden=256, n_classes=7).to(DEVICE)
audio_model.load_state_dict(torch.load('audio_net_best.pt', map_location=DEVICE))
audio_model.eval()

mfcc_transform = torchaudio.transforms.MFCC(sample_rate=AUDIO_SR,
                                            n_mfcc=40,
                                            melkwargs={'n_fft':512,
                                                       'hop_length':256,
                                                       'n_mels':64})

def record_audio(duration=AUDIO_DUR, sr=AUDIO_SR):
    """Returns a torch Tensor of shape (1, N)"""
    raw = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    wave = torch.from_numpy(raw.squeeze().astype(np.float32))
    return wave.unsqueeze(0)

def logits_to_proba(logits):
    return torch.nn.functional.softmax(logits, dim=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open webcam')

print('🚀  Press "q" in the video window to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
    detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(48, 48))

    audio_wave = record_audio(duration=AUDIO_DUR, sr=AUDIO_SR)

    with torch.no_grad():
        audio_logits = audio_model(audio_wave.to(DEVICE))
        audio_proba = logits_to_proba(audio_logits)

    for (x, y, w, h) in detections:
        face_crop = gray[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_img = Image.fromarray(face_crop)
        face_tensor = face_transform(face_img).unsqueeze(0).to(DEVICE)  # type: ignore  # (1,1,48,48)

        with torch.no_grad():
            face_logits = face_model(face_tensor)
            face_proba = logits_to_proba(face_logits)

        fused_proba = fuse_probs(face_proba, audio_proba, weight_face=0.5)

        pred_idx = int(fused_proba.argmax(dim=1).item())
        label = EMOTION_LABELS[pred_idx]
        confidence = float(fused_proba[0, pred_idx].item())

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame,
                    f'{label}:{confidence:.2f}',
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255), 2)

    cv2.imshow('Multimodal Emotion Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
