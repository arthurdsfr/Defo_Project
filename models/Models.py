import os
import sys
import torch
import random
import numpy as np
from torch import nn
from itertools import chain
from progress.bar import Bar
from torchvision import datasets
from torchvision import transforms as T
from sklearn.metrics import confusion_matrix
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils

from models.Decoder import *
from models.Featureextractor import *
from utils.tools import *
from utils.CustomLosses import *

# Set random seed for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

# Call the function to set the seed
set_random_seed(1234)



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


class Models():
    def __init__(self, args, dataset):
        self.args = args
        self.args.input_nc = dataset.num_channels
        
        self.deforestation_classes = dataset.num_classes
        self.args.output_nc= dataset.num_classes
            
        # Feature extractor definition
        if 'dino' in self.args.featureextractor_arch:
            self.feature_extractor = DinoFeaturizer(self.args)
            print(f"Model feature extractor {args.featureextractor_arch} built.")
            self.args.image_nc = self.args.input_nc
            self.args.input_nc = self.feature_extractor.n_feats
            #Decoder definition
            if 'unetr' in self.args.segmentationhead_arch:
                self.segmentation_head = UneTrDecoder(self.args)
                print(f"Model Segmentation Head {args.segmentationhead_arch} built.") 
            
        if 'deeplab' in self.args.featureextractor_arch:
            self.feature_extractor = DeepLabFeaturizer(self.args)
            print(f"Model feature extractor {args.featureextractor_arch} built.")
            self.segmentation_head = DeepLabDecoder(self.args)
            print("Model Segmentation Head built.")
        
        if 'vnet' in self.args.featureextractor_arch:
            self.feature_extractor = VnetFeaturizer(self.args)
            print(f"Model feature extractor {args.featureextractor_arch} built.")
            self.segmentation_head = VnetDecoder(self.args)
            print("Model Segmentation Head built.")
    
        
        print("Segmentation Head params:", sum(p.numel() for p in self.segmentation_head.parameters()))
        print("Feature Extractor params:", sum(p.numel() for p in self.feature_extractor.parameters()))
        
        self.feature_extractor.cuda()
        self.segmentation_head.cuda()
        
          
        if self.args.phase == 'train':
            self.train_loader = torch.utils.data.DataLoader(dataset.train_dataset,
                                                            shuffle = True,
                                                            batch_size=self.args.batch_size_per_gpu,
                                                            num_workers=self.args.num_workers,
                                                            pin_memory=True,
                                                        )
            self.val_loader = torch.utils.data.DataLoader(dataset.valid_dataset,
                                                          shuffle = True,
                                                          batch_size=self.args.batch_size_per_gpu,
                                                          num_workers=self.args.num_workers,
                                                          pin_memory=True,
                                                        )
            
            
            
            print(f"Data loaded with {len(dataset.train_dataset)} train and {len(dataset.valid_dataset)} val imgs.")

            
            self.class_weights = dataset.weights.copy()
            self.class_weights_ = torch.FloatTensor(dataset.weights).cuda()
            if self.args.cost_function == "cross_entropy":
                self.Loss = nn.CrossEntropyLoss(weight = self.class_weights_, ignore_index = dataset.ignore_index, reduction='mean')
            if self.args.cost_function == "focal_loss":
                self.Loss = FocalLoss(self.class_weights_, gamma=2, ignore_index = dataset.ignore_index, size_average = True)
            if self.args.optimizer == "adamw":
                self.optimizer = torch.optim.AdamW(chain(self.segmentation_head.parameters(),
                                                            self.feature_extractor.parameters()), 
                                                self.args.lr,
                                                betas=(0.9, 0.999), 
                                                eps=1e-08, 
                                                weight_decay=0.00004,
                                                )
            elif self.args.optimizer == "adam":
                self.optimizer = torch.optim.Adam(chain(self.segmentation_head.parameters(),
                                                        self.feature_extractor.parameters()), 
                                                self.args.lr,
                                                betas=(0.9, 0.999), 
                                                eps=1e-08, 
                                                weight_decay=0.00004,
                                                )

            if self.args.scheduler_type == "cosine":
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=0)
            elif self.args.scheduler_type == "poly":
                self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=4, power=0.9)
                    


        elif self.args.phase == 'test':
            self.test_loader = torch.utils.data.DataLoader(dataset.test_dataset,
                                                           batch_size = self.args.batch_size_per_gpu,
                                                           num_workers = self.args.num_workers,
                                                           pin_memory = True,
                                                        )
            print(f"Data loaded with {len(dataset.test_dataset)} test imgs.")
    
    def train(self):
        epoch = 0
        BestEpoch = 0
        Patience = 0
        BestF1_Validation = 0

        Train_Loss = []
        Valid_Loss = []
        Train_Ac   = []
        Valid_Ac   = []
        Train_Pr   = []
        Valid_Pr   = []
        Train_Re   = []
        Valid_Re   = []
        Train_F1   = []
        Valid_F1   = []

        if self.args.training_iterations == 0:
            self.args.training_iterations = len(self.train_loader)
        if self.args.validati_iterations == 0:
            self.args.validati_iterations = len(self.val_loader)            

        self.feature_extractor.eval()
        while epoch < self.args.epochs:
            Acc_t = []
            F1_t  = []
            Rec_t = []
            Pre_t = []
            Acc_v = []
            F1_v  = []
            Rec_v = []
            Pre_v = []
            print("-"*100)
            print(f"Epoch {epoch}/{self.args.epochs}")
            print("Training...")

            f = open(self.args.checkpoints_savepath + "Log.txt", "a")
            f.write("-" * 50 + "\n")
            f.write(f"Epoch {epoch}/{self.args.epochs}\n")
            f.write("Training...\n")
            
            
            self.segmentation_head.train()
            if not self.args.freeze_featureextractor:
                self.feature_extractor.train()

            epoch_loss = []
            batch_counter = 0
            cm_total_tr = np.zeros((self.deforestation_classes, self.deforestation_classes))
            cm_total_vl = np.zeros((self.deforestation_classes, self.deforestation_classes))
            
            
            bar =  Bar('Processing', max = self.args.training_iterations)
            for data in self.train_loader:
                self.optimizer.zero_grad()
                self.target = data['labels'].cuda(non_blocking=True)
                self.mask = data['mask'][:,0,:,:].cuda(non_blocking=True)
                if np.sum(self.mask.cpu().numpy()) != 0:
                    if self.args.featureextractor_arch == 'dino':
                        feature_d = self.feature_extractor(data['images'].cuda(non_blocking=True), n = self.args.dino_intermediate_blocks)
                    else:
                        feature_d = self.feature_extractor(data['images'].cuda(non_blocking=True))
                    
                    logits, self.predictions = self.segmentation_head(feature_d)
                    
                    d_loss = self.Loss(logits, self.target)
                    loss = d_loss
                    
                    ###################################"Preforming the backward step"#################################################
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss.append(loss.item())
                    ##################################"Computing training metrics"####################################################
                    if batch_counter % 1 == 0:
                        y_predic_ = np.argmax(self.predictions.detach().cpu().numpy(), 1)
                        data["pred"] = y_predic_
                        y_target_ = self.target.cpu().numpy()
                        mask_ = self.mask.cpu().numpy()
                        y_predic_ = y_predic_.reshape((y_predic_.shape[0] * y_predic_.shape[1] * y_predic_.shape[2], 1))
                        y_target_ = y_target_.reshape((y_target_.shape[0] * y_target_.shape[1] * y_target_.shape[2], 1))
                        mask_     = mask_.reshape((mask_.shape[0] * mask_.shape[1] * mask_.shape[2], 1))
                    
                    
                        available_pixels = np.transpose(np.array(np.where(mask_ == 1)))
                        y_target = y_target_[available_pixels[:,0], available_pixels[:,1]]
                        y_predic = y_predic_[available_pixels[:,0], available_pixels[:,1]]   
                        
                        
                        cm_total_tr += confusion_matrix(y_target, y_predic, labels=np.arange(self.deforestation_classes))
                        
                        
                    batch_counter += 1
                    if batch_counter > self.args.training_iterations:
                        break  
                
                bar.next()
            bar.finish()
            print("Computing Classification Score...")
            results = Compute_Metrics_CF(cm_total_tr)
            Loss_ = (np.sum(epoch_loss))/len(epoch_loss)
            Acc = results["Acc"]
            Pre = results["Pre"]
            Rec = results["Rec"]
            F1 = results["F1"]
            MI = results["mIoU"]
            print("Task Classification results:")
            print(f"epoch {epoch} average loss: {Loss_:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f} mIoU: {MI:.2f}")
            f.write("Task Classification results:\n")
            f.write(f"epoch {epoch} average loss: {Loss_:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f} mIoU: {MI:.2f}\n")
            
            #Storing the training results
            Train_Loss.append(Loss_)
            Train_Ac.append(Acc/100)
            Train_Pr.append(Pre/100)
            Train_Re.append(Rec/100)
            Train_F1.append(F1/100)
            
            
            print("Validating...")
            f.write("Validating...\n")
            F1 = []
            self.feature_extractor.eval()
            self.segmentation_head.eval()
            epoch_loss = []
            if "multitask" in self.args.task:
                self.segmentation_head_aux_t.eval()
                ct_epoch_loss = []
                if "full" in self.args.task:
                    self.segmentation_head_aux_s.eval()
                    cs_epoch_loss = []
     
            batch_counter = 0
            bar =  Bar('Processing', max = self.args.validati_iterations)
            for data in self.val_loader:
                with torch.no_grad():
                    #self.target = data['target_hot'].cuda(non_blocking=True)
                    self.target = data['labels'].cuda(non_blocking=True)
                    self.mask = data['mask'][:,0,:,:].cuda(non_blocking=True)
                    if np.sum(self.mask.cpu().numpy()) != 0:
                        #weights = np.ones((self.target.size(dim = 0), self.target.size(dim = 1), self.target.size(dim = 2), self.target.size(dim = 3)))
                        #for i in range(len(self.class_weights)):
                        #    weights[:, i, :, :] = self.class_weights[i] * weights[:,i,:,:]
                        #self.class_weights_t = torch.from_numpy(weights).cuda(non_blocking=True)
                    
                        if self.args.featureextractor_arch == 'dino':
                            feature_d = self.feature_extractor(data['images'].cuda(non_blocking=True), n = self.args.dino_intermediate_blocks)
                        else:
                            feature_d = self.feature_extractor(data['images'].cuda(non_blocking=True))
                
                        logits, self.predictions = self.segmentation_head(feature_d)


                        y_predic_ = np.argmax(self.predictions.detach().cpu().numpy(), 1)
                        y_target_ = self.target.cpu().numpy()
                        mask_ = self.mask.cpu().numpy()
                        y_predic_ = y_predic_.reshape((y_predic_.shape[0] * y_predic_.shape[1] * y_predic_.shape[2], 1))
                        y_target_ = y_target_.reshape((y_target_.shape[0] * y_target_.shape[1] * y_target_.shape[2], 1))
                        mask_     = mask_.reshape((mask_.shape[0] * mask_.shape[1] * mask_.shape[2], 1))
                
                
                        available_pixels = np.transpose(np.array(np.where(mask_ == 1)))
                        y_target = y_target_[available_pixels[:,0], available_pixels[:,1]]
                        y_predic = y_predic_[available_pixels[:,0], available_pixels[:,1]]

                    
                        #loss = self.Loss()
                        loss = self.Loss(logits, self.target)
                        epoch_loss.append(loss.item())
                        #Computing the metrics
                        cm_total_vl += confusion_matrix(y_target, y_predic, labels=np.arange(self.args.num_classes))
                        #Acc_b, F1_b, Rec_b, Pre_b = Compute_Metrics(y_target, y_predic, 'macro')
                        
                        #Acc_v.append(Acc_b), F1_v.append(F1_b), Rec_v.append(Rec_b), Pre_v.append(Pre_b)
                        if batch_counter > self.args.validati_iterations:
                            break  
                        batch_counter += 1
                
                bar.next()
            bar.finish()
            # Computing the training loss and metrics
            results = Compute_Metrics_CF(cm_total_vl)
            
            Loss_ = (np.sum(epoch_loss))/len(epoch_loss)
            Acc = results["Acc"]
            Pre = results["Pre"]
            Rec = results["Rec"]
            F1_class = results["F1"]
            MI = results["mIoU"]

            print(f"epoch {epoch} average loss: {Loss_:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1_class:.2f} mIoU: {MI:.2f}")
            f.write(f"epoch {epoch} average loss: {Loss_:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1_class:.2f} mIoU: {MI:.2f}\n")
            
            #Storing the training results
            Valid_Loss.append(Loss_)
            Valid_Ac.append(Acc/100)
            Valid_Pr.append(Pre/100)
            Valid_Re.append(Rec/100)
            Valid_F1.append(F1_class/100)

            F1.append(F1_class)
            

            F1 = np.sum(F1)/len(F1)
            if F1 > BestF1_Validation:
                BestF1_Validation = F1
                BestEpoch = epoch
                Patience = 0

                torch.save(self.segmentation_head.state_dict(), self.args.checkpoints_savepath + "segmentation_head.pth")
                if not self.args.freeze_featureextractor:
                    torch.save(self.feature_extractor.state_dict(), self.args.checkpoints_savepath + "feature_extractor.pth")
                print(f"Best model saved with F1 score: {F1: .2f}")
                f.write(f"Best model saved with F1 score: {F1: .2f}\n")
            else:
                Patience += 1
                if Patience > self.args.patience:
                    print(f"Resume: Training finished with model's F1-score: {BestF1_Validation:.2f} obtained at epoch: {BestEpoch}")
                    f.write(f"Resume: Training finished with model's F1-score: {BestF1_Validation:.2f} obtained at epoch: {BestEpoch}\n")
                    f.close()
                    break
            epoch += 1
            self.scheduler.step()
            f.close()

            if self.args.training_graphs:
                createplot(Train_Loss, 
                        Valid_Loss, 
                        np.arange(len(Train_Loss)), 
                        self.args.checkpoints_savepath,
                        "Cross Entropy Loss")
                
                createplot(Train_F1, 
                        Valid_F1, 
                        np.arange(len(Train_F1)), 
                        self.args.checkpoints_savepath,
                        "F1 Score")

                createplot(Train_Re, 
                        Valid_Re, 
                        np.arange(len(Train_Re)), 
                        self.args.checkpoints_savepath,
                        "Recall")

                createplot(Train_Pr, 
                        Valid_Pr, 
                        np.arange(len(Train_Pr)), 
                        self.args.checkpoints_savepath,
                        "Precision")
                
                createplot(Train_Ac, 
                        Valid_Ac, 
                        np.arange(len(Train_Ac)), 
                        self.args.checkpoints_savepath,
                        "Accuracy")
        
    def evaluate(self):

        self.feature_extractor.eval()
        self.segmentation_head.eval()

        #hmresults_path = self.args.results_savepath + '/heat_maps_'   + self.args.eval_type + '/'
        if "domains" in self.args:
            prresults_path = self.args.results_savepath + '/predictions_' + self.args.eval_type + "_" + self.args.domains + '/'
        else:
            prresults_path = self.args.results_savepath + '/predictions_' + self.args.eval_type + '/'
        
        #os.makedirs(hmresults_path, exist_ok=True)
        os.makedirs(prresults_path, exist_ok=True)
        bar =  Bar('Predicting test images', max = len(self.test_loader))
        counter = 0
        for data in self.test_loader:
            if self.args.eval_type == "center_crop":
                self.singlesample_estimator(counter, data, prresults_path)
            elif self.args.eval_type == "sliding_windows":
                self.slidingwindows_estimator(counter, data, prresults_path)
            
            counter += 1
            bar.next()
        bar.finish()
            
    def slidingwindows_estimator(self, counter, data, prresults_path):
        image = data["images"]
        image_org = data["images_org"]
        label = data["labels"][0,0,:,:].numpy()[:,:,np.newaxis]                
        mask = data["mask"][0,0,:,:].numpy()[:,:,np.newaxis]
        image_info = data["image_info"]
        image_name = image_info["image_path"][0].split("/")[-1][:-4]
        image_info = image_info["image_info"]
        coordinates = image_info["Coordinates"][0, :, :]
        k1 = image_info["k1"].cpu().numpy()[0]
        k2 = image_info["k2"].cpu().numpy()[0]
        stride = image_info["stride"].cpu().numpy()[0]
        overlap = image_info["overlap"].cpu().numpy()[0]
        step_row= image_info["step_row"].cpu().numpy()[0]
        step_col= image_info["step_col"].cpu().numpy()[0]
        if self.args.procedure == "non_overlapped_center":
            heat_map_ = np.zeros((image.shape[2], image.shape[3], self.args.output_nc))
        elif self.args.procedure == "average_overlap":
            heat_map_ = torch.zeros((image.shape[2], image.shape[3], self.args.output_nc)).cuda()
            logits_sum = torch.zeros((image.shape[2], image.shape[3], self.args.output_nc)).cuda()
            logits_cou = torch.zeros((image.shape[2], image.shape[3])).cuda()
        
        for i in range(coordinates.shape[0]):
            y_min = coordinates[i,0].to(torch.int)
            y_max = coordinates[i,2].to(torch.int)
            x_min = coordinates[i,1].to(torch.int)
            x_max = coordinates[i,3].to(torch.int)
            image_patch = image[:,:,y_min:y_max,x_min:x_max]

            with torch.no_grad():
                if self.args.featureextractor_arch == 'dino':
                    feature = self.feature_extractor(image_patch.cuda(non_blocking=True), n = self.args.dino_intermediate_blocks)
                else:
                    feature = self.feature_extractor(image_patch.cuda(non_blocking=True))
                
                logits, predictions = self.segmentation_head(feature)
                predictions = predictions.cpu().numpy()
                
                if self.args.procedure == "non_overlapped_center":
                    for c in range(self.args.output_nc):
                        heat_map_[y_min.numpy() : y_min.numpy() + stride, x_min.numpy() : x_min.numpy() + stride, c] = predictions[0, c, overlap//2 : overlap//2 + stride, 
                                                                                                                                            overlap//2 : overlap//2 + stride]
                elif self.args.procedure == "average_overlap": 
                    for c in range(self.args.output_nc):
                        logits_sum[y_min : y_max, x_min : x_max, c] += logits[0, c, :, :]
                        logits_cou[y_min : y_max, x_min : x_max] += 1

        if self.args.procedure == "average_overlap":
            for c in range(self.args.output_nc):
                heat_map_[:,:,c] = torch.div(logits_sum[:,:,c], logits_cou)
            
            heat_map_ = heat_map_.softmax(2).detach().cpu().numpy()
        
        image_org = image_org[0,:,:,:]
        heat_map = heat_map_[: k1 * stride - step_row, : k2 * stride - step_col, :]
        predictions = np.argmax(heat_map, 2)
        predictions = predictions[:,:,np.newaxis]
        predictions = np.concatenate((image_org, predictions, label, mask, heat_map[:, :, 1][:,:,np.newaxis]), axis = 2)
        np.save(prresults_path + image_name, predictions)             

    def singlesample_estimator(self, counter, data, prresults_path):
        image = data["images"]
        image_org = data["images_org"]
        label = data["labels"][0,0,:,:].numpy()[:,:,np.newaxis]             
        mask = data["mask"][0,0,:,:].numpy()[:,:,np.newaxis]
        image_info = data["image_info"]
        image_name = image_info["image_path"][0].split("/")[-1][:-4]
        image_info = image_info["image_info"]
        coordinates = image_info["Coordinates"][0, :, :]
        
        
        y_min = coordinates[0,0].to(torch.int)
        y_max = coordinates[0,2].to(torch.int)
        x_min = coordinates[0,1].to(torch.int)
        x_max = coordinates[0,3].to(torch.int)
        image_patch = image[:,:,y_min:y_max,x_min:x_max]
        image_org = image_org[0, y_min:y_max,x_min:x_max,:]
        label_patch = label[y_min:y_max,x_min:x_max,:]
        mask_patch = mask[y_min:y_max,x_min:x_max,:]
        with torch.no_grad():
            if self.args.featureextractor_arch == 'dino':
                feature = self.feature_extractor(image_patch.cuda(non_blocking=True), n = self.args.dino_intermediate_blocks)
            else:
                feature = self.feature_extractor(image_patch.cuda(non_blocking=True))
            
            logits, predictions = self.segmentation_head(feature)
            predictions = predictions.permute(0, 2, 3, 1).cpu().numpy()[0,:,:,:]
        
        #image_org = unnorm(image_patch).permute(0, 2, 3, 1).cpu().numpy()[0,:,:,:]
        
        predictions = np.argmax(predictions, 2)
        predictions = predictions[:,:,np.newaxis]
        predictions = np.concatenate((image_org, predictions, label_patch, mask_patch), axis = 2)
        
        np.save(prresults_path + image_name, predictions)




