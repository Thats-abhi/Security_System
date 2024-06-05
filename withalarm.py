import cv2
from djitellopy import tello
import cvzone
##from email.mime.multipart import MIMEMultipart
##from email.mime.text import MIMEText
##from email.mime.base import MIMEBase
##from email import encoders
##import smtplib
from playsound import playsound

thres = 0.55
nmsThres = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

##me = tello.Tello()
##me.connect()
##print(me.get_battery())
##me.streamoff()
##me.streamon()
##
##me.takeoff()
##me.move_up(80)


while True:
    success, img = cap.read()
    #img = me.get_frame_read().frame
##    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
##    try:
##        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
##            cvzone.cornerRect(img, box)
##            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
##                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
##                        1, (0, 255, 0), 2)
##            if classNames[classId - 1] == 'cell phone':
##                print("cellphon")
##                playsound('alarm.wav')
##          #  if classIds.flatten() == 76:
##            #    print("cell phon detect ")
##    except:
##        pass
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1] == 'cell phone':
                cvzone.cornerRect(img, box)
                cv2.putText(img, f'cell phone {round(conf * 100, 2)}%',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
                playsound('alarm.wav')
##                cv2.imwrite("NewPicture.jpg",img)
##
##            #cv2.imshow('image', img)
##            fromaddr = "vikas105106@gmail.com"
##            password = "qzhgjvvdlqgorzsx"
##            toaddr = "karann2207@gmail.com"
##
##            msg = MIMEMultipart()
##
##            msg['From'] = fromaddr
##            msg['To'] = toaddr
##            msg['Subject'] = "hii this is mail"
##            body = "Body_of_the_mail"
##            msg.attach(MIMEText(body, 'plain'))
##            filename = "NewPicture.jpg"
##            attachment = open(filename, "rb")
##            p = MIMEBase('application', 'octet-stream')
##            p.set_payload((attachment).read())
##            encoders.encode_base64(p)
##            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
##            msg.attach(p)
##            server = smtplib.SMTP('smtp.gmail.com', 587)
##            server.starttls()
##            server.login(fromaddr,password)
##            text = msg.as_string()
##            server.send_message(msg)
##            server.quit()
        
    except:
        pass

    #me.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
