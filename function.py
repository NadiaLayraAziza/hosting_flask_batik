import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import pandas as pd
import math
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

def ekstraksi_glcm(image_input):
    
    ## Grayscale with matrix multiply
    w = np.array([[[ 0.1141, 0.5870, 0.2989]]])
    grayscale = cv2.convertScaleAbs(np.sum(image_input*w, axis=2))

    # Contras Enhance for grayscale
    gray_img_eqhist = cv2.equalizeHist(grayscale)
    clahe=cv2.createCLAHE(clipLimit=40)
    gray_img_clahe=clahe.apply(gray_img_eqhist)

    data = gray_img_clahe

    def derajat0 (img):
        max = np.max(img)
        imgTmp = np.zeros([max+1, max+1])
        for i in range (len(img)):
            for j in range (len(img[i])-1):
                imgTmp[img[i,j], img[i,j+1]] += 1

        transpos = np.transpose(imgTmp)
        data = imgTmp+transpos

        tmp = 0
        for i in range (len(data)):
            for j in range (len(data)):
                tmp+=data[i,j]

        for i in range (len(data)):
            for j in range (len(data)):
                data[i,j]/=tmp
        return data

    def derajat45 (img):
        max = np.max(img)
        imgTmp = np.zeros([max+1, max+1])
        for i in range (len(img)-1):
            for j in range (len(img[i])-1):
                imgTmp[img[i+1,j], img[i,j+1]] += 1

        transpos = np.transpose(imgTmp)
        data = imgTmp+transpos

        tmp = 0
        for i in range (len(data)):
            for j in range (len(data)):
                tmp+=data[i,j]

        for i in range (len(data)):
            for j in range (len(data)):
                data[i,j]/=tmp
        return data

    def derajat90 (img):
        max = np.max(img)
        imgTmp = np.zeros([max+1, max+1])
        for i in range (len(img)-1):
            for j in range (len(img[i])):
                imgTmp[img[i+1,j], img[i,j]] += 1

        transpos = np.transpose(imgTmp)
        data = imgTmp+transpos

        tmp = 0
        for i in range (len(data)):
            for j in range (len(data)):
                tmp+=data[i,j]

        for i in range (len(data)):
            for j in range (len(data)):
                data[i,j]/=tmp
        return data

    def derajat135 (img):
        max = np.max(img)
        imgTmp = np.zeros([max+1, max+1])
        for i in range (len(img)-1):
            for j in range (len(img[i])-1):
                imgTmp[img[i,j], img[i+1,j+1]] += 1

        transpos = np.transpose(imgTmp)
        data = imgTmp+transpos

        tmp = 0
        for i in range (len(data)):
            for j in range (len(data)):
                tmp+=data[i,j]

        for i in range (len(data)):
            for j in range (len(data)):
                data[i,j]/=tmp
        return data

    hasil = []
    dat=[]
    dat.append(derajat0 (data))
    dat.append(derajat45 (data))
    dat.append(derajat90 (data))
    dat.append(derajat0 (data))
    hasil.append(dat) 

    def entropy (data):
        entro = 0
        for i in range (len (data)):
            for j in range (len (data)):
                if data[i,j] > 0.0:
                    entro += -(data[i,j] * math.log(data[i,j]))
        return entro

    x = ['0', '45', '90', '135']
    data0 = []
    data45 = []
    data90 = []
    data135 = []
    hasilnya = []
    hasilnya_bin = []

    for j in tqdm (range(len(hasil)), desc="Ekstraksi"):
        da = []
        for i in hasil[j]:
            den = entropy (i)
            da.append(den)
        hasilnya.append(da)

    namatabel = ['entropy_0', 'entropy_45', 'entropy_90', 'entropy_135']
    df_ent_new = pd.DataFrame(hasilnya, columns=namatabel)

    # ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
    def calc_glcm_all_agls(img, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    # def calc_glcm_all_agls(img, label, desc, props, dists=[1], agls=[0], lvl=256, sym=True, norm=True):
        
        glcm = graycomatrix(img, 
                            distances=dists, 
                            angles=agls, 
                            levels=lvl,
                            symmetric=sym, 
                            normed=norm)
        
        feature = []
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
        for item in glcm_props:
                feature.append(item)
        # for ent in entropy:
        #         feature.append(ent) 
        
        return feature


    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
    # properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    properties = ['correlation', 'homogeneity', 'contrast', 'energy']

    glcm_all_agls = []
    glcm_value = (calc_glcm_all_agls(gray_img_clahe, props=properties))
    glcm_all_agls.append(glcm_value)
    
    columns = []
    angles = ['0', '45', '90', '135']
    # angles = ['0']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)

    glcm_df_old = pd.DataFrame(glcm_all_agls, columns = columns)
    # glcm_df.head()
    glcm_df = pd.concat([df_ent_new, glcm_df_old], axis=1, join='inner')
    return glcm_df