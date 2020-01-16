import torchvision.transforms as Tr
from config import CUSTOM_LABELS
import random
import cv2
import matplotlib.pyplot as plt

def get_prediction(img, threshold, model):

    transform = Tr.Compose([Tr.ToTensor()])
    img = transform(img)
    model.eval()
    with torch.no_grad():
        pred = model([img.to(device)])

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    # masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [CUSTOM_LABELS[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    # masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(img, img_numpy, model, threshold=0.5, rect_th=1, text_size=0.7, text_th=2):
    boxes, pred_cls = get_prediction(img, threshold, model)

    img = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    for i in range(len(pred_cls)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(255, 255, 255), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,255),thickness=text_th)
        
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()