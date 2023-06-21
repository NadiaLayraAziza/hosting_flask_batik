from flask import Flask, request
from flask import render_template, url_for
from sqlalchemy import create_engine, text
from PIL import Image
import MySQLdb.cursors
import os
import pickle
from function import ekstraksi_glcm
from werkzeug.utils import secure_filename
import cv2

model = pickle.load(open('model_SVM_GLCM.sav', 'rb'))
sc = pickle.load(open('sc_SVM_GLCM.sav', 'rb'))

app = Flask(__name__)

db_url = "mysql://root:nadia123@localhost/batik"
engine = create_engine(db_url, pool_size=5, pool_recycle=3600)
conn = engine.connect()

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route("/")
def home():
    return render_template("index.html", active="home")
    
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    for f in os.listdir(app.config['UPLOAD']):
        os.remove(os.path.join(app.config['UPLOAD'], f))
    f = request.files['file']
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
    basepath, app.config['UPLOAD'], secure_filename(f.filename))
    f.save(file_path)
    namafile = secure_filename(f.filename)
    image = Image.open(file_path)
    image.thumbnail((128, 128))
    image.save("static/uploads/upload_resized.jpg")
    img = cv2.imread("static/uploads/upload_resized.jpg")
    fitur_glcm = ekstraksi_glcm(img)
    df_transformed = sc.transform(fitur_glcm)
    pred = model.predict(df_transformed)[0]
    values = {'nama': pred}
    sql_text = text("""SELECT * FROM master_data WHERE nama = :nama""")
    result = conn.execute(sql_text, values)
    detail = result.fetchone()
    return render_template("ResultUpload.html", namafile=namafile, predictions=pred, detail=detail)
    

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #fs = request.files['snap'] # it raise error when there is no `snap` in form
        fs = request.files.get('snap')
        if fs:
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
            basepath, 'static', secure_filename('image.jpg'))
            fs.save(file_path)
            print('FileStorage:', fs)
            print('filename:', fs.filename)
            return 'Capture Success!'
        else:
            return 'Capture Failed!'
    
    return 'Hello World!'

@app.route('/prediction_webcam')
def predict_camera():
    image = Image.open("static/image.jpg")
    # (left, top, right, bottom)
    cropped_image = image.crop((80, 0, 560, 480))
    cropped_image.save("static/cropped-image.jpg")
    image_resized = Image.open("static/cropped-image.jpg")
    image_resized.thumbnail((128, 128))
    image_resized.save("static/webcam_resized.jpg")
    img = cv2.imread("static/webcam_resized.jpg")
    fitur_glcm = ekstraksi_glcm(img)
    df_transformed = sc.transform(fitur_glcm)
    pred = model.predict(df_transformed)[0]
    
    values = {'nama': pred}
    sql_text = text("""SELECT * FROM master_data WHERE nama = :nama""")
    result = conn.execute(sql_text, values)
    detail = result.fetchone()
    return render_template("ResultWebcam.html", predictions=pred, detail=detail)

@app.route("/MasterData")
def MasterData():
    sql_text = text("select * from master_data")
    result = conn.execute(sql_text)
    data = result.fetchall()
    return render_template("MasterData.html", active="MasterData", data=data)

@app.route("/MasterData/<id>", methods=["GET"])
def show(id):
    values = {'id': id}
    sql_text = text("""SELECT * FROM master_data WHERE id = :id""")
    result = conn.execute(sql_text, values)
    detail = result.fetchone()
    images = os.listdir(os.path.join(app.static_folder, "batik-images/" + detail[1]))
    return render_template("detail.html", active="detail", detail=detail, images=images)

if __name__ == "__main__":
    app.run(debug = True)