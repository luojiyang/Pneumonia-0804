from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
import cv2 #remove for cloud
from PIL import Image #use PIL
import numpy as np
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        file.save("./static/"+filename) #remove when cloud
        file = open("./static/"+filename,"r") #remove when cloud
        model = load_model("Pneumonia")
        image = cv2.imread("./static/"+filename) #remove for cloud
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #remove for cloud
        img = cv2.merge([gray,gray,gray]) #remove for cloud
# for cloud add this (cloud need to use PIL instead of CV2): image = Image.open(file)
# for cloud add this: img = np.asarray(image)
        img.resize((150,150,3))
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshape
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        pred=model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))
if __name__ == "__main__":
    app.run()
