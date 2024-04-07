from flask import Flask, render_template, Response
import cv2

app=Flask(__name__)


specs_ori = cv2.imread('glass.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


def capture_by_frames(): 
    global camera
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #frame = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0, (120, 120), (350, 350))
         #Draw the rectangle around each face
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

                transparentOverlay(face_glass_roi_color, specs)
                transparentOverlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Default Index
@app.route('/')
def index():
    return render_template('index.html')

# POST start request
@app.route('/start',methods=['POST'])
def start():
    return render_template('index.html')

# POST stop request
@app.route('/stop',methods=['POST'])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template('stop.html')

# Video Capture
@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=False,use_reloader=True, port=8000)