
import cv2 
import numpy as np
img=cv2.imread("9.jpg")

img_width=img.shape[1]
img_height=img.shape[0]  
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

labels=["cember"]
colors=np.random.uniform(0,255,size=(1,2))


model=cv2.dnn.readNetFromDarknet("E:/AI/YOLO/cember_train/darknet/cember-yolov4.cfg", "cember-yolov4_last.weights") 
layers=model.getLayerNames()  

output_layers=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]   
model.setInput(img_blob)
detection_layers=model.forward(output_layers)

idsList=[]
boxesList=[]
confidenceList=[]

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores=object_detection[5:]  
        predicted_id=np.argmax(scores) 
        confidence=scores[predicted_id]  
        
        if confidence > 0.30:
            label=labels[0]
            bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height]) 

            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
            start_x=int(box_center_x - (box_width/2))
            start_y=int(box_center_y - (box_height/2))
             
            idsList.append(predicted_id)
            confidenceList.append(float(confidence))
            boxesList.append([start_x,start_y,int(box_width),int(box_height)])

            

maxids=cv2.dnn.NMSBoxes(boxesList,confidenceList,0.5,0.4)
for maxid in maxids:
    maxClassID=maxid
    box=boxesList[maxClassID]
    start_x=box[0]
    start_y=box[1]
    box_width=box[2]
    box_height=box[3]
    predicted_id=idsList[maxClassID]
    label=labels[0]
    confidence=confidenceList[maxClassID]

        
    end_x=start_x + box_width
    end_y=start_y + box_height
    
    box_color=colors[0]
                
                
    label="{}: {:.2f}%".format(label, confidence*100) 
    print("Predicted object {}".format(label)) 
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2)
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,1.5,box_color,1)
            
cv2.imshow("Detection Window",img)            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            



