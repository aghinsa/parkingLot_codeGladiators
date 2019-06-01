import os 
import numpy as np
import cv2
from imageio import imread,imwrite
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

class PostProcess:
    def __init__(self,confidence_threshold,nms_threshold):
        self.confidence_threshold=confidence_threshold
        self.nms_threshold=nms_threshold
    def initModel(self,model_config,model_weights):
        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model Initialised")
        
    def _getOutputsNames(self,net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    def drawBox(self,image,box):
        """
        box=[class,top_left_x,top_left_y,bottom_right_x,bottom_right_y]
        if class 0 : not_occupied, green box
                     occupied, red box
                    mark centre in yellow
        """
        _class,top_x,top_y,w,h=box
        bottom_x = top_x + w
        bottom_y = top_y - h
        
        # drawing box
        _RED=(255,0,0)
        _GREEN=(0,255,0)
        _YELLOW=(255,255,0)
        _THICKNESS=2
    
        if _class == 0:
            cv2.rectangle(image,(top_x,top_y),(bottom_x,bottom_y),_GREEN,_THICKNESS)
        elif _class == 1:
            cv2.rectangle(image,(top_x,top_y),(bottom_x,bottom_y),_RED,_THICKNESS)
    
        # drawing circle
        _RADIUS=3
        cx= int((top_x+bottom_x)/2)
        cy= int((top_y+bottom_y)/2)
        cv2.circle(image,(cx,cy),_RADIUS,_YELLOW,thickness=2)
        return
        
        
        
    def cleanUpAndDraw(self,image,net_outs,draw=True):
        """
        input : image,net_outputs
        Does ouput thresholding and NonMax supression
        """
        image_height,image_width=image.shape[:2]
        class_ids=[]
        confidences=[]
        boxes=[]

        for outs in net_outs:
            for prediction in outs:
                class_scores=prediction[5:]
                predicted_class=np.argmax(class_scores)
                class_score=class_scores[predicted_class]
                
                #confidence thresholding
                if prediction[4]>self.confidence_threshold:
                    if class_scores[predicted_class]>self.confidence_threshold:
                        cx=int(prediction[0] * image_width)
                        cy=int(prediction[1] * image_height)
                        w=int(prediction[2] * image_width)
                        h=int(prediction[3] * image_height)
                        
                        top_left_x=int(cx-w/2)
                        top_left_y=int(cy+h/2)
                        boxes.append([top_left_x,top_left_y,w,h])
                        confidences.append(float(class_scores[predicted_class])) # this needs to be float for NMSBOXes to work
                        class_ids.append(predicted_class)
                        
        #non max supression
        indices=cv2.dnn.NMSBoxes(boxes,confidences,self.confidence_threshold,self.nms_threshold)
        image_boxes=[]
        
        if(draw):
            img=image.copy()
        for i in indices:
            i=i[0]
            box=[class_ids[i],*boxes[i]]
            image_boxes.append(box)
            if(draw):
                self.drawBox(img,box)
                
        if(draw):    
            return image_boxes,img
        else:
            return image_boxes,None
    
    def processBatch(self,image_paths,save_dir="-1"):
        """
        saves to ./'predictions'
        """
        all_boxes=[]
        
        if save_dir=="-1":
            save=False
        else:
            save=True
            
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for image_path in tqdm(image_paths):
            img=imread(image_path)
            blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), [0,0,0], 1, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self._getOutputsNames(self.net))
            image_boxes,img=self.cleanUpAndDraw(img,outs,draw=save)
            filename=image_path.split('/')[-1].rstrip(".jpg")
            if(save):
                save_dir=save_dir.rstrip("/")
                imwrite(save_dir+'/'+filename+'_predict.jpg',img)
                print(f"{filename} saved")
            all_boxes.append(image_boxes)
        return all_boxes
    
    def processImage(self,image,save_path="-1"):
        img=image.copy()
        blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), [0,0,0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self._getOutputsNames(self.net))
        image_boxes,img=self.cleanUpAndDraw(img,outs)
        if save_path!="-1":
            imwrite(save_path,img)
            print(f"Saved to {save_path}")
        return image_boxes,img


##Tunable Params
confidence_threshold=0.5
nms_threshold=0.5
model_config=os.path.abspath('./parkLot_yolov3.cfg')
model_weights=os.path.abspath('./final_weights/parkLot_yolov3_7400.weights')

worker=PostProcess(confidence_threshold=confidence_threshold,nms_threshold=nms_threshold)
worker.initModel(model_config,model_weights)


## cml parsing
parser = argparse.ArgumentParser()
parser.add_argument("--image",default="-1")
parser.add_argument("--images_dir",default="-1")
parser.add_argument("--save_dir",default="-1")
args=parser.parse_args()

if args.image!='-1':
    img=imread(args.image)
    worker.processImage(img,save_path="./parkLotOut.jpg")
if args.images_dir!='-1':
    images_dir_parent=os.path.abspath(args.images_dir)
    images_list=[ images_dir_parent+'/'+ temp_image for temp_image in os.listdir(images_dir_parent) ]
    worker.processBatch(image_paths=images_list,save_dir=args.save_dir)





        
                        
                        
                        