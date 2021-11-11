from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='../front', static_folder='../static')

app.config["IMAGE_UPLOADS"] = "/home/nick/py-progs/portfolio/Pic2Tab web app/back/user_uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG", "JPG", "JPEG", "GIF"]
app.config['MAX_IMAGE_SIZE'] = 10*1024*1024 #10 MB

#Function that checks if image has an allowed extension
def CheckAllowedExtension(img_filename):
    if not "." in img_filename:
        return False
    img_extension = img_filename.rsplit(".", 1)[1]
    if img_extension.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/', methods = ["GET", "POST"])
def UploadFile():

    if request.method == "POST":
        #Checks if any files have been uploaded
        if request.files:
            image = request.files["image"]

            #Checks if uplaoded image exceeds maximum file size
            image.seek(0, os.SEEK_END)
            if image.tell() > app.config['MAX_IMAGE_SIZE']:
                print("Image too large")
                return redirect(request.url)
            image.seek(0)

            #Checks if uploaded image has a name
            if image.filename == "":
                print("Image must have a filename")
                return redirect(request.url)

            #Checks if uploaded image has allowed extension
            if not CheckAllowedExtension(image.filename):
                print("That image extension is not allowed")
                return redirect(request.url)
            else:
                #Create secure name for the uploaded image
                image_filename_secure = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], image_filename_secure))

            print(f"{image_filename_secure} saved")

            return redirect(request.url)

    return render_template('main.html')