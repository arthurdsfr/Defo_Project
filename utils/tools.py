import os
import cv2
import numpy as np
import seaborn as sns
import torch
from torchvision import transforms as T
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def ReferenceColormap(index, dataset):
    if dataset == 'sistemas':
        colors = [(0, 0, 0),
                (123, 213, 207),
                (229, 114, 238),
                (198, 21, 21),
                (218, 71, 209),
                (232, 234, 87),
                (19, 27, 129),
                (233, 117, 2),
                (208, 77, 9),
                (68, 231, 134),
                (255, 255, 255)
                ]
    elif dataset == 'cityscapes':
        colors = [(128, 64, 128),
                  (244, 35, 232),
                  #(250, 170, 160),
                  #(230, 150, 140),
                  (70, 70, 70),
                  (102, 102, 156),
                  (190, 153, 153),
                  (180, 165, 180),
                  #(150, 100, 100),
                  #(150, 120, 90),
                  (153, 153, 153),
                  #(153, 153, 153),
                  (250, 170, 30),
                  (220, 220, 0),
                  (107, 142, 35),
                  (152, 251, 152),
                  (70, 130, 180),
                  (220, 20, 60),
                  (255, 0, 0),
                  (0, 0, 142),
                  (0, 0, 70),
                  (0, 60, 100),
                  #(0, 0, 90),
                  #(0, 0, 110),
                  (0, 80, 100),
                  (0, 0, 230),
                  (119, 11, 32),
                  ]
    elif dataset == 'sistemasuff':
        colors = [(0, 0, 0),
                  (250, 0, 0),# Equipamanto
                  (0, 255, 0),# Escadas
                  (0, 0, 255),# Estruturas
                  (255, 153, 0), # Guardacorpo
                  (255, 102, 204), # Piso
                  (0, 51, 0), # Suportes
                  (102, 51, 0), # TVF
                  (102, 0, 102), # Teto
                  (0, 0, 0), # Background when used in training
                  ]
    elif dataset == "ifremer":
        colors = [(  0,   0,   0),    # Substratum
                  (250, 102, 255),    # B. azoricus + microbial mats
                  (  0, 255,   0),    # Bathymodiolus azoricus
                  (  0,   0, 255),    # Blue glass sponge
                  (255, 153,  50),    # Cataetyx laticeps
                  (255, 102, 204),    # Deposits
                  (  0,  51,   0),    # Microbial mat
                  (255, 153,   0),    # Orange
                  (102,   0, 102),    # Pedonculate
                  ( 68, 231, 134),    # Porifera / Hydrozoa / Foraminifera ?
                  (255,   0,   0),    # Red pelagic shrimp
                  (153,   0, 204),    # Segonzacia mesatlantica
                  ( 51,  51,   0),    # White
                  (153,  51, 102),    # Zoantharia
                 ]
    elif dataset == "ifremer8":
        colors = [(  0,   0,   0),    # Substratum
                  (250, 102, 255),    # B. azoricus + microbial mats
                  (  0, 255,   0),    # Bathymodiolus azoricus
                  (  0,   0, 255),    # Blue glass sponge
                  (255, 153,  50),    # Cataetyx laticeps
                  #(255, 102, 204),    # Deposits
                  (  0,  51,   0),    # Microbial mat
                  (255, 153,   0),    # Orange
                  #(102,   0, 102),    # Pedonculate
                  #( 68, 231, 134),    # Porifera / Hydrozoa / Foraminifera ?
                  #(255,   0,   0),    # Red pelagic shrimp
                  #(153,   0, 204),    # Segonzacia mesatlantica
                  #( 51,  51,   0),    # White
                  #(153,  51, 102),    # Zoantharia
                 ]
    elif dataset == "deforestation":
        colors = [(  0,   0,   0), # No Deforestation
                  (255,   0,   0)  # Deforestation
        ]
    
    return colors[index]

def Compute_Metrics(true_labels, predicted_labels, average, zero_division=0):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average = average, zero_division = zero_division)
    recall = 100*recall_score(true_labels, predicted_labels, average = average, zero_division = zero_division)
    precision = 100*precision_score(true_labels, predicted_labels, average = average, zero_division = zero_division)
    return accuracy, f1score, recall, precision

def Compute_Metrics_CF(cm):
    
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    acc = diagonal_sum/sum_of_all_elements

    Pr_c = np.diag(cm)/cm.sum(axis = 0)
    Re_c = np.diag(cm)/cm.sum(axis = 1)

    F1_c = (2 * Pr_c * Re_c)/(Pr_c + Re_c)
    F1_c[np.isnan(F1_c)] = 0
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(cm)  #  Take the value of the diagonal element , Returns a list of
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)  
    # axis = 1 Represents the value of the confusion matrix row , Returns a list of ;
    # axis = 0 Means to take the value of the confusion matrix column , Returns a list of
    IoU_class = intersection / union  #  Returns a list of , Its value is... Of each category IoU
    IoU_class[np.isnan(IoU_class)] = 0
    mIoU = 100 * np.mean(IoU_class)  #  Find each category IoU The average of
    
    Pr_c[np.isnan(Pr_c)] = 0
    Re_c[np.isnan(Re_c)] = 0

    Acc = 100 * acc
    F1 = np.mean(100 * F1_c)#(np.sum(F1_t))/len(F1_t)
    Pre = np.mean(100 * Pr_c)#(np.sum(Pre_t))/len(Pre_t)
    Rec = np.mean(100 * Re_c)#(np.sum(Rec_t))/len(Rec_t)
    

    return {"Acc": Acc, "F1": F1, "Pre": Pre, "Rec": Rec, "mIoU": mIoU}
    
     

def createplot(yt, yv, x, savepath, title):

    plt.figure()

    plt.plot(x, yt, color = 'red', label = 'train')
    plt.plot(x, yv, color = 'blue',label = 'validation')
    plt.xlabel("Number of epochs")
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(savepath + title + ".jpg")
    plt.close()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], np.round(y[i], 2), ha = 'center')

def createbarplot(df, savepath, title):
    plt.figure(figsize=(10,6))
    sns.barplot(data = df, x = "Metrics", y = "AVG Scores")
    plt.errorbar(df['Metrics'].values, df['AVG Scores'].values, df['STD'].values, fmt = 'ko')
    addlabels(df['Metrics'].values, df['AVG Scores'].values)
    plt.ylim(0, 1)
    plt.show()
    plt.xticks(rotation = 45)
    plt.savefig(savepath + title + ".jpg")
    plt.close()

def get_metrics_fth(df):

    y_true = df["TrueLabels"].values
    y_pred = df["Predictions"].values

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1= f1_score(y_true, y_pred)

    return {'Accuracy': accuracy , 'Recall': recall, 'Precision': precision, 'F1score': f1}

def Visualize_Predictions(Predictions, path, dataset, background):
    alpha = 0.5
    predictions_image = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    groundtruth_image = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    t_classes = np.unique(Predictions[:,:,4])
    p_classes = np.unique(Predictions[:,:,3])
    
    for c in p_classes:
        if c != 255:
            if background:
                color = ReferenceColormap(int(c), dataset)
            else:
                color = ReferenceColormap(int(c + 1), dataset)
        c_indexs = np.transpose(np.array(np.where((Predictions[:,:,3] == c) & (Predictions[:,:,5] == 1))))
        predictions_image[c_indexs[:,0], c_indexs[:,1], 0] = color[0]
        predictions_image[c_indexs[:,0], c_indexs[:,1], 1] = color[1]
        predictions_image[c_indexs[:,0], c_indexs[:,1], 2] = color[2]
    for c in t_classes:
        if c != 255:
            if background:
                color = ReferenceColormap(int(c), dataset)
            else:
                color = ReferenceColormap(int(c + 1), dataset)
        c_indexs = np.transpose(np.array(np.where((Predictions[:,:,4] == c) & (Predictions[:,:,5] == 1))))
        groundtruth_image[c_indexs[:,0], c_indexs[:,1], 0] = color[0]
        groundtruth_image[c_indexs[:,0], c_indexs[:,1], 1] = color[1]
        groundtruth_image[c_indexs[:,0], c_indexs[:,1], 2] = color[2]

    #image_1 = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    #image_2 = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))

    image_1 = cv2.addWeighted(Predictions[:,:,:3].astype(np.uint8), 1 - alpha, predictions_image.astype(np.uint8), alpha, 0)
    image_2 = cv2.addWeighted(Predictions[:,:,:3].astype(np.uint8), 1 - alpha, groundtruth_image.astype(np.uint8), alpha, 0)

    if Predictions[:,:,3].shape[0] >= Predictions[:,:,3].shape[1]:
        f, ax = plt.subplots(1,2)
    else:
        f, ax = plt.subplots(2,1)
    plt.axis('off') 
    ax[0].imshow(image_2)
    ax[0].axis('off')
    ax[0].set_title("Original Image + Ground Truth")
    
    ax[1].imshow(image_1)
    ax[1].axis('off')
    ax[1].set_title("Original Image + Predictions")
    
    #ax[2].imshow(predictions_image/255)
    #ax[2].axis('off')
    #ax[2].set_title("Predicted Image")
    
    plt.savefig(path, dpi=1000, bbox_inches='tight')
    plt.close()

    if Predictions[:,:,3].shape[0] >= Predictions[:,:,3].shape[1]:
        f, ax = plt.subplots(1,3)
    else:
        f, ax = plt.subplots(3,1)
    plt.axis('off') 
    ax[0].imshow(Predictions[:,:,:3].astype(np.uint8))
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    
    ax[1].imshow(groundtruth_image.astype(np.uint8))
    ax[1].axis('off')
    ax[1].set_title("Ground Truth")
    
    ax[2].imshow(predictions_image.astype(np.uint8))
    ax[2].axis('off')
    ax[2].set_title("Predicted Image")
    
    plt.savefig(path[:-4] + "_.jpg", dpi=1000, bbox_inches='tight')
    plt.close()

def Visualize_trainingsample(sample, path, dataset, background):
    image = unnorm(sample["images"]).permute(0, 2, 3, 1).cpu().numpy()[0,:,:,:3]
    label = sample["labels"][0,:,:].numpy()[:,:,np.newaxis]                
    mask = sample["mask"][0,0,:,:].numpy()[:,:,np.newaxis]
    pred = sample["pred"][0,:,:][:,:,np.newaxis]
    #image_name = sample["image_info"]['image_name'][0]
    #coords = sample["image_info"]["coords"][0].numpy()
    #pad = sample["image_info"]["pad_tuple"][0]
    Predictions = np.concatenate((image, mask, label, pred), axis = 2)

    groundtruth_image = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    predicted_image = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    mask_image = np.zeros((Predictions.shape[0], Predictions.shape[1], 3))
    for i in range(3):
        mask_image[:,:,i] += 255 * Predictions[:,:,3]
    t_classes = np.unique(Predictions[:,:,4])
    p_classes = np.unique(Predictions[:,:,5])
    
    
    for c in t_classes:
        if c != 255:
            if background:
                color = ReferenceColormap(int(c), dataset)
            else:
                color = ReferenceColormap(int(c + 1), dataset)
            c_indexs = np.transpose(np.array(np.where(Predictions[:,:,4] == c)))
            groundtruth_image[c_indexs[:,0], c_indexs[:,1], 0] = color[0]
            groundtruth_image[c_indexs[:,0], c_indexs[:,1], 1] = color[1]
            groundtruth_image[c_indexs[:,0], c_indexs[:,1], 2] = color[2]

    for c in p_classes:
        if background:
            color = ReferenceColormap(int(c), dataset)
        else:
            color = ReferenceColormap(int(c + 1), dataset)
        c_indexs = np.transpose(np.array(np.where(Predictions[:,:,5] == c)))
        predicted_image[c_indexs[:,0], c_indexs[:,1], 0] = color[0]
        predicted_image[c_indexs[:,0], c_indexs[:,1], 1] = color[1]
        predicted_image[c_indexs[:,0], c_indexs[:,1], 2] = color[2]

    f, ax = plt.subplots(2,2)
    plt.axis('off') 
    ax[0,0].imshow(Predictions[:,:,:3])
    ax[0,0].axis('off')
    ax[0,0].set_title("Original Image")
    ax[0,1].imshow(mask_image/255)
    ax[0,1].axis('off')
    ax[0,1].set_title("Mask")
    ax[1,0].imshow(groundtruth_image/255)
    ax[1,0].axis('off')
    ax[1,0].set_title("Ground Truth")
    ax[1,1].imshow(predicted_image/255)
    ax[1,1].axis('off')
    ax[1,1].set_title("Predicted image")

    plt.savefig(path + '.jpg', dpi=1000, bbox_inches='tight')
    plt.close()

def mask_creation(mask_row, mask_col, num_patch_row, num_patch_col, Train_tiles, Valid_tiles, Undesired_tiles):
    train_index = 1
    teste_index = 2
    valid_index = 3
    undesired_index = 4
    
    patch_dim_row = mask_row//num_patch_row
    patch_dim_col = mask_col//num_patch_col
    
    mask_array = 2 * np.ones((mask_row, mask_col))
    
    train_mask = np.ones((patch_dim_row, patch_dim_col))
    valid_mask = 3 * np.ones((patch_dim_row, patch_dim_col))
    undesired_mask = 4 * np.ones((patch_dim_row, patch_dim_col))
    counter_r = 1
    counter = 1
    for i in range(0, mask_row, patch_dim_row): 
        for j in range(0 , mask_col, patch_dim_col):           
            train = np.size(np.where(Train_tiles == counter),1)
            valid = np.size(np.where(Valid_tiles == counter),1)
            undesired = np.size(np.where(Undesired_tiles == counter), 1)
            if train == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = train_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = np.ones((mask_row - i, patch_dim_col))
            if valid == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = valid_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 3 * np.ones((mask_row - i, patch_dim_col))
            if undesired == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = undesired_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 4 * np.ones((mask_row - i, patch_dim_col))
            
            counter += 1       
        counter_r += 1
    return mask_array