
# Install required packages
from distutils.dir_util import copy_tree
from re import S
copy_tree("../CapstoneProject/results/lib", "../CapstoneProject/Working/")
#get_ipython().system('pip install efficientnet_pytorch')

import streamlit as st
from PIL import Image
import torch
#from torch.utils.data import DataLoader
import numpy as np
from urllib.request import urlopen
import albumentations as A

import os
#import csv
import albumentations as A
#from glob import glob
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import sys
sys.path.append("../CapstoneProject/results/lib")

import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlopen
from PIL import Image

import skimage.io

import torch
from torch.utils.data import DataLoader


from data import CLASSES, format_prosthesis_name
#from metrics import print_metrics
#from plots import plot_confusion_matrix
from datasets import OrthonetClassificationDataset
from models import get_unet, CLASSIFIER_MODEL_GENERATORS
from training import load_classifier_transforms
#from testing import eval_seg, eval_unetseg, eval_ensemble, write_predictions_to_csv

import temp_img

# Paths
CSV_TEST = "../CapstoneProject/archive/test.csv"
MODEL_DIR = "../CapstoneProject/Orthonet_Models"
DATA_PATH = "../CapstoneProject/archive/orthonet data/orthonet data new"



def prediction(image):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BS_TEST = 4
    N_WORKERS = 1


    _, test_transforms = load_classifier_transforms()
    ds_test = OrthonetClassificationDataset('test', CSV_TEST, DATA_PATH, test_transforms)
    #dl_test = DataLoader(ds_test, BS_TEST, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

    ds_test.stats()

   
    model_2d = CLASSIFIER_MODEL_GENERATORS["efficientnet"](n_in=1, n_out=len(CLASSES)).to(DEVICE)
    model_2d.load_state_dict(torch.load("C:/Users/aksha/Desktop/CapstoneProject/Working/efficientnet_215_0.0944893.pt",map_location=torch.device('cpu'))['state_dict'])
    model_2d.eval()


    unet_path = "C:/Users/aksha/Desktop/CapstoneProject/Working/seg_unet_192_0.9000270.pt"
    unet_model = get_unet(1, 1).to(DEVICE)
    unet_model.load_state_dict(torch.load(unet_path,map_location=torch.device('cpu'))['state_dict'])
    unet_model = unet_model.eval()
    clahe_transform = A.Compose([A.CLAHE(p=1)])



    model_seg = CLASSIFIER_MODEL_GENERATORS["wideresnet50"](n_in=2, n_out=len(CLASSES)).to(DEVICE)
    model_seg.load_state_dict(torch.load("C:/Users/aksha/Desktop/CapstoneProject/Working/wideresnet50.pt",map_location=torch.device('cpu'))['state_dict'])
    model_seg.to(DEVICE).eval()


    # Load the image
    s = image
    img = np.array(Image.open(s))


    # Process image to pass into network
    x = test_transforms(image=img)['image'] / 255.
    x = torch.from_numpy(x).float().unsqueeze_(0).unsqueeze_(0).to(DEVICE)


    with torch.no_grad():
        # Simple classifier
        y_pred_classifier = model_2d(x)
        
        mask = torch.sigmoid(unet_model(x)) > 0.5
        x_m = x.clone()
        x_m[~mask] = 0
        x = torch.cat((x_m, x), dim=1)
        
        # Contrast-limited adaptive histogram equalisation
        x[0,0] = torch.tensor(clahe_transform(image=(x[0,0].cpu().numpy() * 255).astype(np.uint8))['image']/255).float()
                    
        # Seg classifier
        y_pred_segclassifier = model_seg(x)
        
        # Ensemble
        ensemble_weight = 1
        y_pred = ((torch.softmax(y_pred_segclassifier, dim=1)*ensemble_weight + torch.softmax(y_pred_classifier, dim=1)) / (1+ensemble_weight)).cpu().numpy()


    ####################
    print("success")

    top_3_classes = y_pred[0].argsort()[-3:][::-1]
    top_3_confidences = y_pred[0][top_3_classes]



    # Build up an array of example images
    example_images = {}
    for sample in ds_test.samples:
        example_images[sample['labels']] = sample['filenames']

    fig, axes = plt.subplots(3,3,figsize=(16,16))

    for r in axes:
        for c in r:
            c.axis('off')


    class0_path = os.path.join(DATA_PATH, example_images[CLASSES[top_3_classes[0]]])
    #axes[0,2].imshow(skimage.io.imread(class0_path), cmap='gray')
    #axes[0,2].set_title(f"{CLASSES[top_3_classes[0]]}\n{top_3_confidences[0]*100:.2f}% confident")

    class1_path = os.path.join(DATA_PATH, example_images[CLASSES[top_3_classes[1]]])
    #axes[1,2].imshow(skimage.io.imread(class1_path), cmap='gray')
    #axes[1,2].set_title(f"{CLASSES[top_3_classes[1]]}\n{top_3_confidences[1]*100:.2f}% confident")

    class2_path = os.path.join(DATA_PATH, example_images[CLASSES[top_3_classes[2]]])
    #axes[2,2].imshow(skimage.io.imread(class2_path), cmap='gray')
    #axes[2,2].set_title(f"{CLASSES[top_3_classes[2]]}\n{top_3_confidences[2]*100:.2f}% confident")

    return class0_path,class1_path,class2_path,CLASSES[top_3_classes[0]],CLASSES[top_3_classes[1]],CLASSES[top_3_classes[2]],round(top_3_confidences[0]*100,2),round(top_3_confidences[1]*100,2),round(top_3_confidences[2]*100,2)


import requests
import io
#####################################################################################################################
st.set_page_config(layout="wide")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


st.title("Implant Identifier")

p1 = []
try:
    uploaded_image = st.file_uploader("Choose a file")
    title = st.text_input('Enter Image URL')
    if uploaded_image is not None:
        bytes_data = uploaded_image.getvalue()
        temp_img.byte_storage = bytes_data        
        # bytearray(temp_img.byte_storage)

        a1,a2,a3 = st.columns([3,2,3])
        with a2:
            st.image(temp_img.byte_storage, caption='Uploaded X-ray', use_column_width= True)
        im = Image.open(io.BytesIO(bytearray(temp_img.byte_storage)))
        im.save("C:/Users/aksha/Desktop/CapstoneProject/Temp_image/t.png")
        p1 = prediction("Temp_image/t.png")
        print(p1)
        c1,c12,c2,c23,c3 = st.columns([2,1,2,1,2])
        with c1:
            st.image(p1[0], use_column_width = True)
            st.write("Class = ",p1[3],"\n confidence =",p1[6])
            st.download_button("Download Image", p1[0],file_name='Accurate.png')            
        with c2:
            st.image(p1[1], use_column_width = True)
            st.write("Class = ",p1[4],"\n confidence =",p1[7])
            st.download_button("Download Image", p1[1],file_name="Moderate.png")
        with c3:
            st.image(p1[2], use_column_width = True)
            st.write("Class = ",p1[5],"\n confidence =",p1[8])
            st.download_button("Download Image", p1[2],file_name='Least.png')
    
    
    elif title is not None:
        a1,a2,a3 = st.columns([3,2,3])
        with a2:
            st.image(title, use_column_width=True)
        im = Image.open(requests.get(title, stream=True).raw)
        im.save("C:/Users/aksha/Desktop/CapstoneProject/Temp_image/t.png")

        p1 = prediction("Temp_image/t.png")
        print(p1)
        d1,d12,d2,d23,d3 = st.columns([2,1,2,1,2])
        with d1:
            st.image(p1[0], use_column_width = True)
            st.write("Class = ",p1[3],"\n confidence =",p1[6])
            st.download_button("Download Image", p1[0],file_name='Accurate.png')
        with d2:
            st.image(p1[1], use_column_width = True)
            st.write("Class = ",p1[4],"\n confidence =",p1[7])
            st.download_button("Download Image", p1[1],file_name='Moderate.png')
        with d3:
            st.image(p1[2], use_column_width = True)
            st.write("Class = ",p1[5],"\n confidence =",p1[8])
            st.download_button("Download Image", p1[2],file_name='Least.png')
except:
    pass


