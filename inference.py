import cv2
import torch
from torchvision import transforms
from torch.nn import functional as F
import pyshine as ps

model = torch.load('models/full-vgg16,99%-accuracy.pth')
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
classNames = ['WithMask', 'WithoutMask']
threshold=0.6

def cameraPredict():
    #get video capture object
    capture = cv2.VideoCapture(0)
    while(True):
        ret, frame = capture.read()

        # transforming images 
        frameInput = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameInput = cv2.resize(frame, (128, 128))
        frameInput=transforms.ToTensor()(frameInput)

        #passing images to model
        l = model(transforms.Normalize(*stats,inplace=True)(frameInput).unsqueeze(0))

        #calculating probabilities
        _,pred = torch.max(l,dim=1)
        probs = F.softmax(l, dim=1)
        rltd_probs =probs[0][pred.item()]

        #predicted label
        
        predLabel = classNames[pred]
        print(rltd_probs.item())
        
        # if predLabel == "WithoutMask":
            
        #     frame = cv2.putText(
        #                   img = frame,
        #                   text = "Good Morning",
        #                   org = (20, 50),
        #                   fontFace = cv2.FONT_HERSHEY_DUPLEX,
        #                   fontScale = 1,
        #                   color = (125, 246, 55),
        #                   thickness = 2
        #                 )

        if (rltd_probs.item()>threshold):
            frame =  ps.putBText(frame,f'{str(predLabel)}: {int(rltd_probs*100)}%',text_offset_x=30,text_offset_y=70,
                                            vspace=10,hspace=10, font_scale=1,background_RGB=(0,250,250),
                                            text_RGB=(160,32,240))
            
            #get keyboard input
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            
            cv2.imshow('frame',frame)
    capture.release()
    cv2.destroyAllWindows()

cameraPredict()
