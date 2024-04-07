from fastapi import FastAPI, Response, Request
from fastapi.templating import Jinja2Templates
import cv2
from pydantic import BaseModel
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="templates")

specs_ori = cv2.imread('glass.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)

def transparent_overlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

class Frame(BaseModel):
    frame: bytes

def capture_by_frames(): 
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0, (120, 120), (350, 350))

        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                glass_symin = int(y + 1.5 * h / 5)
                glass_symax = int(y + 2.5 * h / 5)
                sh_glass = glass_symax - glass_symin

                cigar_symin = int(y + 4 * h / 6)
                cigar_symax = int(y + 5.5 * h / 6)
                sh_cigar = cigar_symax - cigar_symin

                face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
                face_cigar_roi_color = frame[cigar_symin:cigar_symax, x:x + w]

                specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
                cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)

                transparent_overlay(face_glass_roi_color, specs)
                transparent_overlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield Frame(frame=frame_bytes)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/video_capture")
async def video_capture():
    return Response(next(capture_by_frames()), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/start")
async def start(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

###############################################################################################

if __name__ == "__main__":
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)