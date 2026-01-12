import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    results = model(img)
    return results

def process_results(results):
    df = results.pandas().xyxy[0]  # Get bounding boxes
    print(df.columns)  # Debugging line
    return df

def save_visualization(img, results, output_path):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for index, row in results.iterrows():
        try:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            img_bgr = cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img_bgr = cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except KeyError as e:
            print(f"KeyError: {e} - please check the column names in the results DataFrame.")
    cv2.imwrite(output_path, img_bgr)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    results = predict_image(img)
    processed_results = process_results(results)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{filename}')
    save_visualization(img, processed_results, output_path)

    return redirect(url_for('result', filename=f'output_{filename}'))

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
