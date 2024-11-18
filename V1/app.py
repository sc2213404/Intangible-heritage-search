from flask import Flask,render_template,request
import io
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        current_file_path = os.path.abspath(__file__)
        current_folder_path = os.path.dirname(current_file_path)
        save_dir = current_folder_path+r'\static\images'
        file_path = os.path.join(save_dir, file.filename)
        file.save(file_path)
        '''
        此处调用图片识别文件,并返回一张图片至 static/images/
        '''
        image_name = file.filename
        return render_template('temp.html',image_name = image_name)
    else:
        return render_template('index.html')  


if __name__ == '__main__':
    app.run()