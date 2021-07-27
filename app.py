from posixpath import dirname
from flask  import Flask, request, render_template,redirect, url_for,abort,send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import os.path
import tempfile
import io
import os
import base64


import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

import cv2
import numpy as np
from PIL import Image



# Here are the 91 classes.
OBJECTS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

OBJECTS_html=['all', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',  'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',  'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',  'dining table', 'toilet',  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    predicted_classes= [(OBJECTS[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes

def draw_box(predicted_classes,image,rect_th= 10,text_size= 3,text_th=3):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """

    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
   
        label=predicted_class[0]
        probability=predicted_class[1]
        box=predicted_class[2]

        cv2.rectangle(img, box[0], box[1],(0, 255, 0), rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,label, box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img,label+": "+str(round(probability,2)), box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    
    return img


#Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image  pre-trained on COCO.
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False


#the function calls Faster R-CNN  model_  but save RAM:
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat 

transform = transforms.Compose([transforms.ToTensor()])


app=Flask(__name__)

dostuff=None
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif','.jpeg']
app.config['UPLOAD_PATH'] = 'uploads'
app.config['OBJECTS_PATH'] = 'objects'


app.config['FILE_PATH']=None
app.config['FILE_NAME']=None
dir_name = os.path.join(app.instance_path)

@app.route('/')
def home():
    #new file that  has  been uploaded 
    files= os.listdir(app.config['UPLOAD_PATH'])
    files=[ file  for file in files  if os.path.splitext(file )[1] in app.config['UPLOAD_EXTENSIONS'] ]
    #files that  has  been uploaded that have been  uploaded 
    object_files=os.listdir(app.config['OBJECTS_PATH'])
    object_files=[ file  for file in object_files  if os.path.splitext(file )[1] in app.config['UPLOAD_EXTENSIONS'] ]


    return render_template('index.html', files=files,objects_list=OBJECTS_html,object_files=object_files)


@app.route('/', methods=['POST'])
def upload_file():
    #file object
    uploaded_file = request.files['file']
    #file name  
    filename= secure_filename(uploaded_file.filename)
    #file extention 
    file_ext = os.path.splitext(filename)[1]

    #check if empty file 
    if filename != '':
        # file path /uploads/filename 

        #check if .jpg, .png, .gif if  not send an error 
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        #send back  to home  agument is the fuction "home"
        #upload file  path 
        #uploaded_file.save(filename)
        file_path=os.path.join(app.config['UPLOAD_PATH'], filename)
        app.config['FILE_NAME']=[filename]

        app.config['FILE_PATH']=file_path
        uploaded_file.save(file_path)
        return redirect(url_for('home'))

@app.route('/find_object', methods=['POST']) 
def find():
    

    
    object=request.form.get("objects")
    
  
  
    half = 0.5
    
    image = Image.open(app.config['FILE_PATH'])
    image.resize( [int(half * s) for s in image.size] ) 
    img = transform(image)
    pred = model(torch.unsqueeze(img,0))   

    if object=='all':
        pred_thresh=get_predictions(pred,threshold=0.97)
        print("------alll-----------",object)

    else:
        pred_thresh=get_predictions(pred,threshold=0.97,objects=object)
        print("------------------",object)
    
    print(pred_thresh)
    
    #draw box on image 
    image=draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1) 
    #save image with box with new name 
    filename, file_extension = os.path.splitext(app.config['FILE_NAME'][0])
    new_file_name=filename+"_object"+file_extension
    new_file_path=os.path.join(app.config['OBJECTS_PATH'],new_file_name)


    #save file
    cv2.imwrite(new_file_path, image)

    if (request.form.get("Find_New")):
        os.remove(app.config['FILE_PATH'])


        return redirect(url_for('home'))
            
    return render_template("find_object.html" ,objects=object,file=new_file_name)      
 #serve these uploade files from  following route
@app.route('/uploads/<filename>')
def upload(filename):
    #get file this is called in index.html 
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

 #serve these  files from  following routey
@app.route('/objects/<filename>')
def upload_objects(filename):
    #get file this is called in index.html 
    return send_from_directory(app.config['OBJECTS_PATH'], filename)



if __name__=="__main__":
    app.run(debug=True)