from typing import Text
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import base64
import geocoder
import os


g = geocoder.ip('me')
h = 24.870899,67.096072
count=0

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
def database():
   binaryFrames = []
   sub=os.listdir("D:\Final Year Project\FLASK WEBCAM\deployment\static\images")
   for frames in sub:
      with open("D:\Final Year Project\FLASK WEBCAM\deployment\static\images\\"+frames, "rb") as img_file:
         binaryFrames.append(base64.b64encode(img_file.read()))




      


   

   db=firestore.client()

   docRef = db.collection('guns').document()
   docRef.set({
   'Status':'Detected','Date Time':datetime.now(),'id':docRef.id
   })
   for i in binaryFrames:

      docRef.collection("frames").add({'Image':str(i.decode('utf-8'))})




  
