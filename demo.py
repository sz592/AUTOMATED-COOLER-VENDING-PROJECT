from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import pygame
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
import pyzbar.pyzbar as pyzbar
from edgetpu.detection.engine import DetectionEngine
from PIL import Image

os.putenv("SDL_VIDEODRIVER","fbcon")
os.putenv("SDL_FBDEV","/dev/fb0")
os.putenv("SDL_MOUSEDEV","/dev/input/touchscreen")
os.putenv("SDL_MOUSEDRV","TSLIB")

GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)


pygame.init()
pygame.mouse.set_visible(False)
size=width,height=320,240
speed=[2,2]
black=0,0,0
blue=(0,0,128)
teal=(0,128,128)

screen=pygame.display.set_mode(size)

flag=True
run2=True
def quit_func(channel):
  global flag,run
  flag=False
  run=False
  run2=False
    
GPIO.add_event_detect(27, GPIO.FALLING, callback=quit_func, bouncetime=300)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.1,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


LABELS = open("yolo-coco/voc.names").read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolo-coco/yolov3-tiny_1900.weights"
configPath = "yolo-coco/yolov3-tiny.cfg"
 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs= PiVideoStream().start()

#input("Press Enter to continue...")
bottle_count=0
c_locked=True
run=True
ranga=25
Blue=[192, 140, 75]
Purple=[167, 100, 131]
Red=[33, 3, 137]
Yellow=[102, 177, 198]
cid=["Red Gatorade","Yellow Gatorade","Pepsi","Peach Pure Leaf Tea"]
start=0
old_bottle_counter=None
bottle_counter={"Red Gatorade":0,"Yellow Gatorade":0, "Pepsi":0, "Peach Pure Leaf Tea":0}
reseter={"Red Gatorade":0,"Yellow Gatorade":0, "Pepsi":0, "Peach Pure Leaf Tea":0}

def coral_tpu():
    labels = {}

    # loop over the class labels file
    for row in open("mobilenet_ssd_v2/coco_labels.txt"):
    # unpack the row and update the labels dictionary
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()

    # load the Google Coral object detection model
    print("[INFO] loading Coral model...")
    model = DetectionEngine("mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")

    # load the input image
    
    # image = cv2.flip(image,0)
    frame = vs.read()
    image = cv2.flip(frame,0)
    image = imutils.resize(image, width=500)
    orig = image.copy()

    # prepare the image for object detection by converting (1) it from
    # BGR to RGB channel ordering and then (2) from a NumPy array to PIL
    # image format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)


    # make predictions on the input image
    print("[INFO] making predictions...")
    start = time.time()
    results = model.DetectWithImage(image, threshold=args["confidence"],
    keep_aspect_ratio=True, relative_coord=False)
    end = time.time()
    print("[INFO] object detection took {:.4f} seconds...".format(
    end - start))
    i = 0
    
    
    bg = cv2.imread("bg.jpg")
    #bg - cv2.resize(bg,(328,2464))
    bg = cv2.flip(bg,0)
    #bg = imutils.resize(bg, width=500)
 

    # prepare the image for object detection by converting (1) it from
    # BGR to RGB channel ordering and then (2) from a NumPy array to PIL
    # image format
    #bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    #bg= Image.fromarray(bg)
    # loop over the results
    for r in results:
    # extract the bounding box and box and predicted class label
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        label = labels[r.label_id]
        if label == "bottle":
                bg = cv2.imread("bg.jpg")
    #bg - cv2.resize(bg,(328,2464))
                bg = cv2.flip(bg,0)
                bottle= orig[startY:endY,startX:endX]
                width = endX-startX
                hight = endY-startY
                width_s = int((endX-startX)/2)
                hight_s = int((endY-startY)/2)
                width_e = width-width_s
                hight_e = hight-hight_s
                i = i+1
                
                #cv2.imshow("image",bottle)
                cv2.imwrite("bb.jpg",bottle)
                
                #bottle= Image.fromarray(bottle)
                bg[166-hight_s:166+hight_e,208-width_s:208+width_e]=bottle
                yolo(bg)
                #bg.paste(bottle,(1000,1000,1000+width,1000+hight))
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                        (225, 150, 150), -1)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                text = "{}: {:.2f}%".format(label, r.score * 100)
                #cv2.putText(orig, text, (startX, y),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #cv2.imshow("1",orig)
    cv2.imwrite("1_1.jpg",orig)
    orig_1 = Image.fromarray(orig)
    #orig_1 = Image.fromarray(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB))
    print("s")
    results = model.DetectWithImage(orig_1, threshold=args["confidence"],keep_aspect_ratio=True, relative_coord=False)
    print("end")
    for r in results:
        print("yolo")
# extract the bounding box and box and predicted class label
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        label = labels[r.label_id]
        if label == "bottle":
                bg = cv2.imread("bg.jpg")
    #bg - cv2.resize(bg,(328,2464))
                bg = cv2.flip(bg,0)
                bottle= orig[startY:endY,startX:endX]
                width = endX-startX
                hight = endY-startY
                width_s = int((endX-startX)/2)
                hight_s = int((endY-startY)/2)
                width_e = width-width_s
                hight_e = hight-hight_s
                i = i+1
                
                #cv2.imshow("image",bottle)
                cv2.imwrite("bb.jpg",bottle)
                
                #bottle= Image.fromarray(bottle)
                bg[166-hight_s:166+hight_e,208-width_s:208+width_e]=bottle
                yolo(bg)
                #bg.paste(bottle,(1000,1000,1000+width,1000+hight))
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                        (225, 150, 150), -1)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                text = "{}: {:.2f}%".format(label, r.score * 100)
    


def take_inventory():
    bottle_counter=reseter.copy()
    coral_tpu()
   
    
def yolo(frame):
  global bottle_count
  global bottle_counter
  #frame = vs.read()
  #frame = imutils.rotate(frame, 180)
  #frame = imutils.resize(frame, width=400)
  mini_frame=frame
  # grab the frame dimensions and convert it to a blob
  (H, W) = frame.shape[:2]
  
  # determine only the *output* layer names that we need from YOLO
  ln = net.getLayerNames()
  ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  
  #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
  #  0.007843, (300, 300), 127.5)
  
  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
  # pass the blob through the network and obtain the detections and
  # predictions
  net.setInput(blob)
  detections = net.forward(ln)

  # initialize our lists of detected bounding boxes, confidences, and
  # class IDs, respectively
  boxes = []
  confidences = []
  classIDs = []
 
  # loop over each of the layer outputs
  for output in detections:
    # loop over each of the detections
    for detection in (output):
      # extract the class ID and confidence (i.e., probability) of
      # the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

      # filter out weak detections by ensuring the `confidence` is
      # greater than the minimum confidence
      if confidence > args["confidence"]:
        print("conf:",confidence)
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        
        # update our list of bounding box coordinates, confidences,
        # and class IDs
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)
        print("confidence: ",confidence)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
          args["threshold"])
        

        if len(idxs)>0:
          print("in")
          # loop over the indexes we are keeping
          for i in idxs.flatten():
            print("CID:",classIDs[i])
            bottle_counter[cid[classIDs[i]]]=1+bottle_counter[cid[classIDs[i]]]
            
        print("time elapsed: ", time.time()-start)
        #cv2.imshow("Image", frame)
        #cv2.waitKey(0)
      for item,count in bottle_counter.items():
        if count>0:
          bottle_count+=1
      

def updateTFT(screen_info):
  global c_locked, white, black, blue
  screen.fill(black)
  font=pygame.font.Font('freesansbold.ttf',12)
  stext=""
  if c_locked:
    stext="Cooler Status: Locked"
  else:
    stext="Cooler Status: Unlocked"
  text=font.render(stext,True, blue)
  textrect=text.get_rect()
  textrect.center=(75,20)
  pygame.draw.rect(screen, teal,(0,0,150,40))
  screen.blit(text,textrect)
  for stext,pos in screen_info.items():
    if len(stext)>35:
      font=pygame.font.Font('freesansbold.ttf',10)
    elif len(stext)>25:
      font=pygame.font.Font('freesansbold.ttf',15)
    else:
      font=pygame.font.Font('freesansbold.ttf',20)
    text=font.render(stext,True, (255,255,255))
    textrect=text.get_rect()
    textrect.center=pos
    screen.blit(text,textrect)
  pygame.display.flip()

def check_customer():
  global run,run2
  global bottle_counter, reseter,old_bottle_counter, c_locked
  c_locked=True
  updateTFT({"Scan QR Code":(160,120)})
  #os.system("fswebcam -d /dev/video1 img4")
  imx = vs.read()
  #t=input("Enter img name: ")
  #imx=cv2.imread(t)
  dec=pyzbar.decode(imx)
  if(len(dec)>0 ):#and (dec[0].data is "Rohit")):
    g=str(dec[0].data)
    g=g[2:len(g)-1]
    print(g)
    name=g
    #a=input("Scan QR code")
    #name="Rohit"
    if(name == "Rohit" or name=="Sizhe" or name=="Joe"):
      print("COOLER UNLOCKED")
      c_locked=False
      x=("Hello "+name+", please open the cooler and select a drink")
      print(x)
      updateTFT({x:(160,120)})
      time.sleep(5)
      return True
    elif (name == "Maintenance"):
      x=("Maintenance in progress, restock drinks now")
      y=("Tap screen when done restocking")
      c_locked=False
      print(x)
      updateTFT({x:(160,100), y:(160,130)})
      run=True
      while run:
        time.sleep(.1)
        if (not GPIO.input(17)):
          print("got press, done restock")
          take_inventory()
          old_bottle_counter=bottle_counter
            #bottle_count=0
          bottle_counter=reseter.copy()
          xp,yp=pygame.mouse.get_pos()
          peps="Pepsi: "+str(old_bottle_counter["Pepsi"])
          rgat="Red Gatorade: "+str(old_bottle_counter["Red Gatorade"])
          ygat="Yellow Gatorade: "+str(old_bottle_counter["Yellow Gatorade"])
          tea="Peach Pure Leaf Tea: "+str(old_bottle_counter["Peach Pure Leaf Tea"])
          c_locked=True
          updateTFT({"INVENTORY: ":(110,60),tea:(160,150),peps:(160,90),rgat:(160,110),ygat:(160,130), "To rescan tap screen, otherwise wait":(160,180)})
          now=time.time()
          run=False
          run2=True
          while(time.time()-now<5 and run2):
            print("waiting for tap or 5 second timeout")
            time.sleep(.1)
            if (not GPIO.input(17)):
              run=True
              run2=False
              c_locked=False
              updateTFT({x:(160,100), y:(160,130)})
          
         # break;
            
    else:
      print("Not recognized")
      c_locked=True
      updateTFT({"User not recognized":(160,120)})
      time.sleep(3)
  return False
    
customer=False

def bounce_pepsi():
  ball=pygame.image.load("/home/pi/FinalDemo/test/pepsi2.png")
  ballrect=ball.get_rect()
  flag2=True
  while flag2:
    for event in pygame.event.get():
      if event.type is pygame.MOUSEBUTTONUP:
        xp,yp=pygame.mouse.get_pos()
        flag2=False
    time.sleep(.03)
    ballrect=ballrect.move(speed)
    if ballrect.left<0 or ballrect.right>width:
      speed[0]=-speed[0]
    if ballrect.top<0 or ballrect.bottom >height:
      speed[1]=-speed[1]
    screen.fill(black)
    screen.blit(ball,ballrect)
    pygame.display.flip()


print("TAKE INVENTORY:")
start=time.time()
take_inventory()
#print(bottle_count, " bottles detected")  
print(bottle_counter)
old_bottle_counter=bottle_counter
bottle_count=0
bottle_counter=reseter.copy()
bounce_pepsi()    
    
while flag:
    
  if customer is not True:
    customer=check_customer()
  else:
    print("TAKE INVENTORY:")
    start=time.time()
    take_inventory()
  
    print("")
    #print(bottle_count, " bottles detected")
  
    print(bottle_counter)
  
    no_purchase=True
    if(old_bottle_counter is not None):
      update_str=""
      cnt = 0
      for drink in cid:
        if(old_bottle_counter[drink] is not bottle_counter[drink]):
          purchased = old_bottle_counter[drink]-bottle_counter[drink]
          if(purchased>0):
            no_purchase=False
            s = (str(purchased)+ " " +str(drink))
            update_str+=s+", "
            print(s)
            print(purchased)
            cnt+=purchased
    if no_purchase:
      c_locked=True
      print("No drink purchased")
      updateTFT({"No drink purchased":(160,120)})
      time.sleep(8)
    else:
      c_locked=True
      updateTFT({update_str[0:len(update_str)-2]+" Purchased":(160,100), "You will be charged $"+ str(1.99*cnt)+", Thank You":(160,140)})
      time.sleep(7)
        
  
    old_bottle_counter=bottle_counter
    bottle_count=0
    bottle_counter=reseter.copy()
    customer=False
  if customer: 
    c_locked=False   
    updateTFT({"Tap Screen when done selecting drink":(160,120)})
    print("ENTER WHEN DONE SELECTING DRINK")
    wait_for_tap=True
    while(wait_for_tap):
      if (not GPIO.input(17)):
        xp,yp=pygame.mouse.get_pos()
        wait_for_tap=False
    
    print("COOLER LOCKED")
    c_locked=True
    #if(a=="q"):
    #  flag=False
    #if(a=="reset"):
    #  bottle_counter=reseter.copy()

vs.stop()

