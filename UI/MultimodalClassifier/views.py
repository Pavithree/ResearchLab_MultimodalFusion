from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

#from keras.models import load_model
from keras.preprocessing import image
import json
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm

import neattext.functions as nfx
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import operator


"""
##uncomment this code to run the prediction using VisualBert Model
###### Start VisualBert Code ######
import glob
import re
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import pickle
import os
import io
import torch
import tensorflow as tf

# import some common detectron2 utilities
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

NUM_OBJECTS = 36

# Load VG Classes
data_path = './models/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes

#Config
cfg = get_cfg()
cfg.merge_from_file("./models/configs/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
predictor = DefaultPredictor(cfg)

def doit(raw_image):
    with torch.no_grad():
        final_array = []
        raw_height, raw_width = raw_image.shape[:2]
        #print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        #print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]
        
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:], 
                score_thresh=0, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
                
        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        final_array.append(roi_features.cpu().numpy())
        final_array.append(instances.pred_boxes.tensor.cpu().numpy())
        final_array.append(raw_height)
        final_array.append(raw_width)
        return final_array

#Feature exrtraction
#feature, bbox, img_w, img_h = doit(image_read)

#VisualBert
VBERT_model = torch.load('./models/VBert/VBertMIMIC.pt');
ftr_r = np.reshape(feature, (1,36,2048))
box_r = np.reshape(bbox, (1,36,4))
bbox = torch.from_numpy(box_r)
feature = torch.from_numpy(ftr_r)
sgmd = torch.nn.Sigmoid()
logit = VBERT_model(feature.cuda(), bbox.cuda(),  report)
result_VBert = sgmd(logit).cpu().numpy();

###### End VisualBert Code ######
##Uncomment this code to run the prediction using VisualBert Model
"""


tokenizer=Tokenizer();

def clean_text(text):
    text_length=[]
    cleaned_text=[]
    for sent in tqdm(text):
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        sent=nfx.remove_stopwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text,text_length


img_height, img_width = 250,300

model_graph = tf.Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('./models/lateFusion/lateFusion.h5')

# Create your views here.

def index(request):
    context = {'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    
    fileObj = request.FILES['filePath']
    fileObjText = request.FILES['filePathText']
    fs = FileSystemStorage()
    imageFilePathName = fs.save(fileObj.name,fileObj)
    imageFilePathName = fs.url(imageFilePathName)
    textFilePathName = fs.save(fileObjText.name,fileObjText)
    textFilePathName =  fs.url(textFilePathName)
    testimage = '.'+imageFilePathName
    testText = '.'+textFilePathName
    img = image.load_img(testimage, target_size=(img_height,img_width))
    
    #os.chdir(textFilePathName)
    with open(testText) as f:
        report = f.readlines()
    #report = "Heart size and pulmonary vascularity appear within normal limits. The lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen. Vascular calcification is noted. No adenopathy is seen. 1. No evidence of active disease.."
    cleaned_report, cleaned_report_length = clean_text(report)
    report_seq=tokenizer.texts_to_sequences(cleaned_report)
    report_pad=pad_sequences(report_seq,maxlen=50)

    #load the nlp model and make the prediction
    model_NLP = tf.keras.models.load_model("./models/NLP")
    result_NLP = model_NLP.predict(report_pad, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,workers=1, use_multiprocessing=False)
    
    #load the visual bert result
    result_VBert = [7.4546150e-04, 5.3781793e-02, 2.7630964e-02, 2.7981797e-02, 2.4510680e-02,
                   1.1066774e-02, 6.5459244e-02, 9.6584296e-01, 2.7130239e-02, 3.0707447e-03,
                  7.0582358e-03, 3.1475675e-01, 7.1236943e-03]

    #late fusion
    input_FeatureVector = result_NLP[0].tolist() + result_VBert
    #input_FeatureVector = result_NLP[0].tolist() + result_VBert.tolist() 
    # if using saved visualBert model, add .tolist() to resut_VBert

    model = load_model('./models/lateFusion/lateFusion.h5')
    y_pred = model.predict([input_FeatureVector])

    #visualisation
    X_labels = ['Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Enlarged Cardiomediastinum',
                'Fracture',
                'Lung Lesion',
                'Lung Opacity',
                'No Finding',
                'Pleural Effusion',
                'Pleural Other',
                'Pneumonia',
                'Pneumothorax']

    probability_illness = list(y_pred.ravel())





    #x= image.img_to_array(img)
    #x=x/255
    #x=x.reshape(1,img_height,img_width,3)
    #with model_graph.as_default():
    #    with tf_session.as_default():
     #       predi= model.predict(x)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    diseases = X_labels
    probability = probability_illness
    ax.bar(diseases,probability)
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
    fig1 = plt.gcf()
    plt.savefig("./media/result.png", bbox_inches = 'tight')
    imageFilePathName = "./media/result.png"

    result = y_pred.tolist()
    index, value = max(enumerate(result[0]), key=operator.itemgetter(1))

    label = X_labels[index]
    #import numpy as np
    #predictedLabel = X_labels[str(np.argmax(y_pred[0]))]

    context = {'imageFilePathName':imageFilePathName, 'textFilePathName': textFilePathName, 'predictedLabel': label}
    return render(request,'index.html',context)