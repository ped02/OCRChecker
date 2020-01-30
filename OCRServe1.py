import numpy as np
import math
import os
import PIL
import cv2
import pytesseract
import re
import io
import base64

import requests

from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import threading
from datetime import datetime

from werkzeug import ImmutableDict
from werkzeug.datastructures import FileStorage

import socket

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('AGG')

curDir = os.getcwd()
pngs = os.path.join(curDir, "Img")

pytesseract.pytesseract.tesseract_cmd = r'/opt/local/bin/tesseract'

matchD = r'\d+\d+\.+\d+\d+\.+\d+\d+\d+\d'
matchMFD = r'[a-zA-Z]+[Ff]+[a-zA-Z]'
matchBBF = r'[a-zA-Z]+[a-zA-Z]+[Ff]'

app = Flask(__name__,static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

hostIP = socket.gethostbyname(socket.gethostname())

def fill(img, r):
    kern1 = np.zeros((2*r+1,1))
    kern1[r:,0] = 1
    
    kern2 = np.zeros((2*r+1,1))
    kern2[:r+1,0] = 1
    
    imgP1 = cv2.morphologyEx(img,cv2.MORPH_DILATE,np.uint8(kern1))
    imgP2 = cv2.morphologyEx(img,cv2.MORPH_DILATE,np.uint8(kern2))
    
    return cv2.bitwise_and(imgP1,imgP2)

def saveComponent(img, fileName):
    sumV = np.sum(img, axis=1)
    sumH = np.sum(img, axis=0)

    fig = plt.figure(constrained_layout=True)
    height = [2, 2]
    width = [2, 4, 2]
    gs = fig.add_gridspec(2, 3, width_ratios=width, height_ratios=height)

    f_ax1 = fig.add_subplot(gs[0, 1:])
    f_ax1.set_title('Img')
    f_ax1.imshow(img)

    f_ax2 = fig.add_subplot(gs[0, :1])
    f_ax2.set_title('Vert')
    f_ax2.plot(sumV,-np.arange(0,img.shape[0]))

    f_ax3 = fig.add_subplot(gs[1, 1:])
    f_ax3.set_title('Hori')
    f_ax3.plot(sumH)

    plt.savefig(os.path.join(pngs,fileName))

def crop(img, vThresh = 0.3, hThresh = 0.3):
    saveComponent(img,"test.png")
    sumV = np.sum(img, axis=1)
    sumH = np.sum(img, axis=0)
    
    sumV = sumV * 1/(np.max(sumV))
    sumH = sumH * 1/(np.max(sumH))

    hIs = -1
    hIe = -1
    
    charHS = -1
    charHE = -1
    
    hSpace = 100
    
    lV = len(sumV)
    lH = len(sumH)
    
    for i in range(lV):
        if(sumV[i] > vThresh):
            charHS = i
            break
        
    for i in range(lV-1,0,-1):
        if(sumV[i] > vThresh):
            charHE = i
            break
        
    for i in range(lH):
        if(sumH[i] > hThresh):
            if(i > hSpace):
                hIs = i - hSpace
            else:
                hIs = 0
            break
        
    for i in range(lH-1,0,-1):
        if(sumH[i] > hThresh):
            if(i + hSpace < lH):
                hIe = i + hSpace
            else:
                hIe = lH
            break
            
    charHeight = charHE-charHS
    scale = 50.0/charHeight
    
    vSpace = math.floor(0.25*charHeight)
    
    print(hIe, hIs)
    print(charHS,charHE,charHeight,scale,vSpace)
    
    if(charHS > vSpace):
        vIs = charHS - vSpace
    else:
        vIs = 0
    
    if(charHE + vSpace < lV):
        vIe = charHE + vSpace
    else:
        vIe = lV
    
    print(math.floor((hIe-hIs)*scale),math.floor((vIe-vIs)*scale))
    return cv2.resize(img[vIs:vIe,hIs:hIe],(math.floor((hIe-hIs)*scale),math.floor((vIe-vIs)*scale)))

def saveImage(img,fileName):
    filePath = os.path.join(pngs,fileName)
    plt.imsave(filePath,img)

def readDates(img):
    #Input IMG : 300 x 3535 x 3
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #saveImage(gray, "Gray.png")

    #mask1 = cv2.bitwise_and(gray,(gray < 150).astype(np.uint8))
    #mask2 = cv2.bitwise_and(mask1,(gray > 15).astype(np.uint8))

    #gray = mask2
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    processed = cv2.morphologyEx(cv2.bitwise_not(gray),cv2.MORPH_ERODE,np.ones((3,3)))
    processed = cv2.GaussianBlur(processed,(5,5),0)
    
    #Tesseract Accept : 50 x 589 x 1 // DOWN SCALE by 6
    resized = crop(processed)
    #resized = crop(cv2.bitwise_not(gray))
    resized = fill(resized,3)
    resized = fill(resized,3)

    #saveImage(resized,"processed.png")

    '''
    testA = []
    for i in range(7,9):
        testRead = []
        testRead.append(pytesseract.image_to_string(resized,lang='DotMatrix_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='5x5_Dots_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='DisplayDots_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='dotOCRDData1',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='Dotrice_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='LCDDot_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='Transit_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='Orario_FT_500',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='dotslayer',config="--psm "+str(i)))
        testRead.append(pytesseract.image_to_string(resized,lang='eng',config="--psm "+str(i)))
        testA.append(testRead)
    print(testA)
    '''

    recognized = pytesseract.image_to_string(resized,lang='LCDDot_FT_500',config='--psm 8')

    print(recognized)
    
    dateSend = searchDate(recognized)

    #RETURN MDF , BBF
    return dateSend

def searchDate(dateString):
    dates = []
    datesIS = []
    datesIE = []

    for i in re.finditer(matchD,dateString):
        dates.append(i.group(0))
        datesIS.append(i.start())
        datesIE.append(i.end())

    #dates = re.findall(matchD,recognized)
    mf = re.search(matchMFD, dateString)
    bf = re.search(matchBBF, dateString)

    print(dates)
    print(datesIS)
    print(datesIE)
    print(mf, bf)
    dateSend = []
    if(len(dates) == 2):
        if(mf is not None and bf is not None):
            if(mf.start() > bf.start()):
                dateSend.append(dates[1])
                dateSend.append(dates[0])
            else:
                dateSend.append(dates[0])
                dateSend.append(dates[1])
        if(mf is not None and bf is None):
            if(mf.start() < datesIS[0] and mf.start() > datesIS[1]):
                dateSend.append(dates[0])
                dateSend.append(dates[1])
            elif(mf.start() > datesIS[0] and mf.start() < datesIS[1]):
                dateSend.append(dates[1])
                dateSend.append(dates[0])
        if(mf is None and bf is not None):
            if(bf.start() < datesIS[0] and bf.start() > datesIS[1]):
                dateSend.append(dates[1])
                dateSend.append(dates[0])
            elif(bf.start() > datesIS[0] and bf.start() < datesIS[1]):
                dateSend.append(dates[0])
                dateSend.append(dates[1])
    elif(len(dates) == 1):
        dateSend.append(dates[0])

    return dateSend

@app.route("/")
def home():
    return render_template("testLinkLaunch.html")
@app.route('/<path:path>')
def send_js(path):
    return send_from_directory(app.static_folder, path)

@app.route('/testConnect', methods=['POST'])
def testConnect():
    if request.method == 'POST':
        testData = request.get_json()

        print(testData)
        testData["Message2"] = "Hello from OCR Server: " + hostIP

        return testData

@app.route('/readImgT', methods=['POST'])
def handleImg():
    if request.method == 'POST':
        #request.files
        data = request.files.to_dict()
        POID = data["POID"]
        imgD = data["IMG"]
        imgN = np.fromstring(imgD.read(),np.uint8)
        imgCV = cv2.imdecode(imgN,cv2.IMREAD_UNCHANGED)
        dates = readDates(imgCV)
        print(dates)
        print("OK")

@app.route('/CheckDates', methods=['POST'])
def handlePOIDImg():
    if request.method == 'POST':

        data = request.get_json()
        #print(data)

        POID = data["POID"]
        imgDat = data["Img"]
        imgStr = base64.b64decode(imgDat)

        filename = os.path.join(pngs,"TestRead.png")
        with open(filename, 'wb') as f:
            f.write(imgStr)

        imageDataP = PIL.Image.open(io.BytesIO(imgStr))
        imageDataN = np.array(imageDataP)
        imageDataC = cv2.imdecode(imageDataN, cv2.IMREAD_UNCHANGED)

        dates = readDates(imageDataC)

        #Send to server to check
        #res = requests.post()

        print(dates)

        return "<html></html>"

@app.route('/getDates', methods=['POST'])
def getDates():
    if request.method == 'POST':

        data = request.get_json()
        #print(data)

        POID = data["POID"]
        imgDat = data["Img"]
        imgStr = base64.b64decode(imgDat)

        print(POID)

        #filename = os.path.join(pngs,"TestRead.png")
        #with open(filename, 'wb') as f:
        #    f.write(imgStr)

        imageDataP = PIL.Image.open(io.BytesIO(imgStr)).convert('RGB')
        imageDataN = np.array(imageDataP, dtype=np.uint8)
        #imageDataC = cv2.imdecode(imageDataN, cv2.IMREAD_UNCHANGED)

        print(imageDataN.shape)
        print(imageDataN)
        #print(imageDataC)

        #saveImage(imageDataN, "TestPng3.png")
        #saveComponent(imageDataN,"f")
        dates = readDates(imageDataN)

        print(dates)

        datesData = dict()
        if(len(dates)==1):
            datesData["EXP"] = dates[0]
        elif(len(dates)==2):
            datesData["MDF"] = dates[0]
            datesData["EXP"] = dates[1]

        return jsonify(datesData)

@app.route('/POIDImg', methods=['POST'])
def Img():
    if request.method == 'POST':

        data = request.get_json()
        #print(data)
        imgDat = data["Img"]
        imgStr = base64.b64decode(imgDat)

        filename = os.path.join(pngs,"TestRead.png")
        with open(filename, 'wb') as f:
            f.write(imgStr)

        #imgW = data["Width"]
        #imgH = data["Height"]

        #imgStD = imgStr.decode()
        #imageio.imread()

        imageDataP = PIL.Image.open(io.BytesIO(imgStr))
        imageDataN = np.array(imageDataP)
        imageDataC = cv2.imdecode(imageDataN, cv2.IMREAD_UNCHANGED)

        #saveComponent(imageDataC,"f")

        #dates = readDates(imageDataC)

        #Extract Data

        #Send to server to check
        #res = requests.post()

        return "<html></html>"

@app.route('/testRead', methods=['POST'])
def testRead():
    if request.method == 'POST':
        data = request.files.to_dict()
        print(data)

        imgSt = data["img"]
        #imgIO = io.StringIO(imgSt.read())

        imgF = imgSt.content_type
        print(imgF)

        imgType = imgF[7:]

        #imageDataP = PIL.Image.open(imgIO).convert('RGB')
        #imageDataN = np.uint8(imageDataP)
        imageDataN = np.fromstring(imgSt.read(), np.uint8)
        print(imageDataN, imageDataN.dtype)
        print(imageDataN.shape)
        #saveImage(imageDataN, "TestPng1.png")
        imageDataC = cv2.imdecode(imageDataN, cv2.IMREAD_UNCHANGED)
        #saveImage(imageDataC, "TestPng2.png")

        #saveComponent(imageDataC,"f")

        dates = readDates(imageDataC)
        print(dates)
        return render_template("testRead.html")

if __name__ == '__main__':
    socketio.run(app,debug=True,host="0.0.0.0",port="8001")
