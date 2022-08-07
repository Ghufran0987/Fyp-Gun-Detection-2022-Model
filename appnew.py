
   
import argparse
from unicodedata import name
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from flask import Flask, render_template, Response
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size,  non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import base64
import geocoder
import os
import pyrebase
import threading




app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')

config={
   "apiKey": "AIzaSyCawM2HsxjqxgIEM2PjNvcfzct1OdWqfeU",
   "databaseURL":"https://console.firebase.google.com/project/fyp-project-27e8c/database/fyp-project-27e8c-default-rtdb/data/~2F",
    "authDomain": "fyp-project-27e8c.firebaseapp.com",
    "projectId": "fyp-project-27e8c",
    "storageBucket": "fyp-project-27e8c.appspot.com",
    "serviceAccount":"fyp-project.json"
    }

cred = credentials.Certificate("fyp-project.json")
firebase_admin.initialize_app(cred)
#db=firestore.client()
#docRef = db.collection('guns').document()

    


def database():
    
    

    binaryFrames = []
    
    
    sub=os.listdir("D:\Final Year Project\FYP 2 Work\Final FYP APPLICATION\Final-Flask-App--master\static\images")
    for frames in sub:
        with open("D:\Final Year Project\FYP 2 Work\Final FYP APPLICATION\Final-Flask-App--master\static\images\\"+frames, "rb") as img_file:
            binaryFrames.append(base64.b64encode(img_file.read()))
    
    
    db=firestore.client()

    docRef = db.collection('guns').document()
    docRef.set({'Status':'Detected','time':datetime.now(),'id':docRef.id})
    
    for i in binaryFrames:
    
        docRef.collection("frames").add({'Image':str(i.decode('utf-8'))})
    







@app.route('/')
def index():
    return render_template('index.html')

def gen():
    source, weights, imgsz, frame_rate = opt.source, opt.weights, opt.img_size, opt.frame_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_logging()
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    cap = cv2.VideoCapture(0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[0, 0, 255], [0, 255, 0]]
    

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    print(f"Video capturing at {cap.get(cv2.CAP_PROP_FPS)} FPS")

   
    
    
   


    is_predicting = False
    count_frame = 1
    count=0
   
    while(True):
        res, img0 = cap.read()
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Pad size
        img = letterbox(img0, imgsz, 32)[0]
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        count_frame += 1
        if (count_frame > frame_rate):
            is_predicting = True
            count_frame = 0

        # string result
        s = f'%gx%g ' % img.shape[2:]

        if is_predicting:
            is_predicting = False
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            count=0
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], img0.shape).round()

                    # Print time (inference + NMS)
                    s += f"| ({t2 - t1:.3f}s) | Fps: {round(1/(t2-t1), 1)}"
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]}'
                        plot_one_box(xyxy, img0, label=label,
                                     color=colors[int(cls)], line_thickness=3)
                        object_name=names[int(cls)]
                        x1=int(xyxy[0].item())
                        y1=int(xyxy[1].item())
                        x2=int(xyxy[2].item())
                        y2=int(xyxy[3].item())
                        original_image=img0
                        cropped_image=img0[y1:y2,x1:x2]

                        while  count<=3:
                            
                            imgNumber = str(count).zfill(5)
                            frameImageFileName = str(f'D:\Final Year Project\FYP 2 Work\Final FYP APPLICATION\Final-Flask-App--master\static\images\{imgNumber}.png')
                            cv2.imwrite(frameImageFileName,original_image)
                            

                            count += 1
                        if(count==3):
                                x=threading.Thread(target=database)
                                x.start()

                        
                        


                                
                            
                        """
                        
                        docRef.set({'Status':'Detected','time':datetime.now(),'id':docRef.id,})
                        
                        sub=os.listdir("D:\FOLDE\Flask-Gun-Detection-Master\static\images")
                        for frames in sub:
                            with open("D:\FOLDE\Flask-Gun-Detection-Master\static\images\\"+frames, "rb") as img_file:
                                binaryFrames.append(base64.b64encode(img_file.read()))

                        
                        
                        for i in binaryFrames:
    
                            docRef.collection("frames").add({'Image':str(i.decode('utf-8'))})

                        """
                        
                        
                        
                        
                        
                            
                         

                            
                            
                            
                            
                        

                        
                        



                        
                        
                        

                        
                        print('detected object name is',object_name)
                        print('Bounding box is',x1,y1,x2,y2)
                        
                        
                        
                        

                        
                        
                        
                       
                        

                            

                            



                        

                    
        # show image
        # cv2.imshow('Number plate detection', img0)
        


        

       
        cv2.imwrite('frame.jpg', img0)
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')
        # String results
        print(s)
        # wait key to break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # Release capture
    cap.release()
    cv2.destroyAllWindows()




@app.route('/video_feed')
def video_feed():
    
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Experiment face mask detection with Yolov5 models')

    parser.add_argument('--host',  type=str,
                        default='192.168.100.4:4000', help="Local IP")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Run app in debug mode")

    parser.add_argument('--weights', nargs='+', type=str,
                        default='./models/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--frame-rate', default=0,
                        type=int, help='sample rate')
    opt = parser.parse_args()

    hostname = str.split(opt.host, ':')
    if len(hostname) == 1:
        port = 4000
    else:
        port = hostname[1]
    host = hostname[0]

    app.run(host=host, port=port, debug=opt.debug, use_reloader=False)
