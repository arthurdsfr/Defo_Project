import os
import sys
import numpy as np
from progress.bar import Bar
import skimage.morphology
from skimage.morphology import square, disk
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.tools import *

class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64)

def compute_ndvi(image_t1, image_t2):
    print("Computing NDVI...")
    ndvi_t1 = np.zeros((image_t1.shape[0] , image_t1.shape[1] , 1))
    ndvi_t2 = np.zeros((image_t2.shape[0] , image_t2.shape[1] , 1))
    ndvi_t1[:,:,0] = np.divide((image_t1[:,:,4] - image_t1[:,:,3]),(image_t1[:,:,4] + image_t1[:,:,3]))
    ndvi_t2[:,:,0] = np.divide((image_t2[:,:,4] - image_t2[:,:,3]),(image_t2[:,:,4] + image_t2[:,:,3]))
    image_t1 = np.concatenate((image_t1, ndvi_t1), axis=2)
    image_t2 = np.concatenate((image_t2, ndvi_t2), axis=2)
    print("NDVI computed and stacked back in the data. Their shape is: " + str(np.shape(image_t1)))
    return image_t1, image_t2
def select_bands(image_t1, image_t2, bands):
    print("Selecting the bands...")
    image_t1_ = np.zeros((image_t1.shape[0], image_t1.shape[1], len(bands)))
    image_t2_ = np.zeros((image_t2.shape[0], image_t2.shape[1], len(bands)))
    counter = 0
    for band in bands:
        image_t1_[:,:,counter] = image_t1[:,:,band]
        image_t2_[:,:,counter] = image_t2[:,:,band]
        counter += 1
    image_t1 = image_t1_.copy()
    image_t2 = image_t2_.copy()
    print("Bands have been selected. Data has the following shape: " + str(np.shape(image_t1)))
    return image_t1, image_t2

def normalize(image_t1, image_t2):
    print('[*]Normalizing the images...')
    scaler = StandardScaler()
    images = np.concatenate((image_t1, image_t2), axis=2)
    images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))
    scaler = scaler.fit(images_reshaped)
    images_normalized = scaler.fit_transform(images_reshaped)
    images = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
    image_t1 = images[:, :, : image_t1.shape[2]]
    image_t2 = images[:, :,   image_t2.shape[2]:]   
    print("Images have been normalized with min and max in: " + str(np.min(image_t1)) + " " +  str(np.min(image_t2)) + "and " + str(np.max(image_t1)) + " " +  str(np.max(image_t2)))
    return image_t1, image_t2, scaler
def compute_difference(self):
    self.difference = self.image_t2 - self.image_t1

class DeforestationDataset():
    def __init__(self, args) -> None:
        self.args = args
        self.weights = [0.4, 2.5]
        self.ignore_index = 255
        
        self.args.domains = self.args.domains[0]
        if self.args.domains == "PA":
            self.dataobject = AMAZON_PA(args)
        if self.args.domains == "RO":
            self.dataobject = AMAZON_RO(args)
        if self.args.domains == "MA":
            self.dataobject = CERRADO_MA(args)

        self.num_classes = len(np.unique(self.dataobject.reference)) - 1
  
        self.Create()

    
    def Create(self):
        if self.args.phase == "train":
            
            print("Creating training dataset")
            self.train_dataset = CustomDatasetClassification(self.dataobject, self.num_classes, 1, self.args)
            print(self.train_dataset.__len__())
            print("Creating validation dataset")
            self.valid_dataset = CustomDatasetClassification(self.dataobject, self.num_classes, 3, self.args)
            print(self.valid_dataset.__len__())
            
            self.num_channels = self.train_dataset.num_channels
        
        elif self.args.phase == 'test' or "visuals" in self.args.phase or "metrics" in self.args.phase:
            self.test_dataset = CustomDatasetClassification(self.dataobject, self.num_classes, 2, self.args)
            self.num_channels = self.test_dataset.num_channels
            self.area_avoided = self.test_dataset.dataobject.area_avoided


class CustomDatasetDomains():
    def __init__(self, dataobject, set, args):
        self.args = args
        self.set = set
        self.tar_transform = ToTargetTensor()
        if set == 1 or set == 3:
            self.train_transforms = T.Compose([T.ToTensor(),
                                            T.RandomApply([
                                                T.RandomHorizontalFlip(p=0.5),
                                                T.RandomVerticalFlip(p=0.5),
                                                #T.RandomRotation(degrees=(0, 180)),
                                            ],p=0.8),
                                            ])
            for i, dictname in enumerate(dataobject):
                print(dictname)
                self.dataobject = dataobject[dictname]
                self.img_rows = self.dataobject.image_t1.shape[0]
                self.img_cols = self.dataobject.image_t1.shape[1]
                if i == 0:
                    self.source = True
                    self.target = False
                    self.source_img_rows = self.dataobject.image_t1.shape[0]
                    self.souce_img_cols = self.dataobject.image_t1.shape[1]
                    self.source_defores_num_classes = len(np.unique(dataobject[dictname].reference)) - 1
                    self.source_cluster_num_classes = len(np.unique(dataobject[dictname].cluster_reference))
                    coordinates_dict = self.Corner_Coordinates_Definition()
                    self.source_coordinates = coordinates_dict["Coordinates"]
                    self.source_images_padded = np.pad(np.concatenate((self.dataobject.image_t1, self.dataobject.image_t2), axis = 2),
                                                       coordinates_dict["pad_tuple_images"], mode = 'symmetric')
                    self.source_num_channels = self.source_images_padded.shape[2]
                    self.source_references_padded = np.pad(self.dataobject.reference, coordinates_dict["pad_tuple_refere"], mode = 'symmetric')
                    self.source_clusterref_padded = np.pad(self.dataobject.cluster_reference, coordinates_dict["pad_tuple_refere"], mode = 'symmetric')

                else:
                    self.target = True
                    self.source = False
                    self.target_img_rows = self.dataobject.image_t1.shape[0]
                    self.target_img_cols = self.dataobject.image_t1.shape[1]
                    self.target_cluster_num_classes = len(np.unique(dataobject[dictname].cluster_reference)) 
                    coordinates_dict = self.Corner_Coordinates_Definition()
                    self.target_coordinates = coordinates_dict["Coordinates"]
                    self.target_images_padded = np.pad(np.concatenate((self.dataobject.image_t1, self.dataobject.image_t2), axis = 2), 
                                                        coordinates_dict["pad_tuple_images"], mode = 'symmetric')
                    self.target_num_channels = self.source_images_padded.shape[2]
                    self.target_clusterref_padded = np.pad(self.dataobject.cluster_reference, coordinates_dict["pad_tuple_refere"], mode = 'symmetric')
                 
            if self.source_coordinates.shape[0] > self.target_coordinates.shape[0]:
                rate = int(self.source_coordinates.shape[0]/self.target_coordinates.shape[0])
                for i in range(rate -1):
                    if i == 0:
                        original_coordinates = self.target_coordinates.copy()
                    self.target_coordinates = np.concatenate((self.target_coordinates, original_coordinates), axis = 0)

            elif self.source_coordinates.shape[0] < self.target_coordinates.shape[0]:
                rate = int(self.target_coordinates.shape[0]/self.source_coordinates.shape[0])
                for i in range(rate-1):
                    if i == 0:
                        original_coordinates = self.source_coordinates.copy()
                    self.source_coordinates = np.concatenate((self.source_coordinates, original_coordinates), axis = 0)

            self.num_channels = self.source_num_channels
            self.s_data_padded = np.concatenate((self.source_images_padded, self.source_references_padded[:,:,np.newaxis], self.source_clusterref_padded[:,:,np.newaxis]), axis = 2)
            self.t_data_padded = np.concatenate((self.target_images_padded, self.target_clusterref_padded[:,:,np.newaxis]), axis = 2)

    def __len__(self):
        if self.args.phase == 'train':
            if self.source_coordinates.shape[0] > self.target_coordinates.shape[0]:
                num_samples = self.target_coordinates.shape[0]
            elif self.source_coordinates.shape[0] < self.target_coordinates.shape[0]:
                num_samples = self.source_coordinates.shape[0]
            return num_samples
        
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        source_coords = self.source_coordinates[idx]
        target_coords = self.target_coordinates[idx]
        
        s_data_patch = self.s_data_padded[int(source_coords[0]) : int(source_coords[2]), int(source_coords[1]) : int(source_coords[3]), :]
        t_data_patch = self.t_data_padded[int(target_coords[0]) : int(target_coords[2]), int(target_coords[1]) : int(target_coords[3]), :]
        s_data_patch = self.train_transforms(s_data_patch)        
        t_data_patch = self.train_transforms(t_data_patch)

        image = s_data_patch[:-2, :, :]
        s_target_1 = s_data_patch[-2, :, :].clone()
        s_target_2 = s_data_patch[-2, :, :].clone()
        c_target = s_data_patch[-1, :, :].clone()
        
        s_target_1[s_target_1 == 255] = -1
        mask = s_target_1 != -1
        mask = mask * 1
        
        t_image = t_data_patch[:-1, :, :]
        t_target = t_data_patch[-1, :, :].clone()
        
        return {"images": image.to(torch.float32), 
                "labels": s_target_2.long(),
                "s_clabels": c_target.long(),
                "mask": mask.unsqueeze(0),  
                "t_images": t_image.to(torch.float32),
                "t_clabels": t_target.long()}

    def Corner_Coordinates_Definition(self):
        # Computing the overlaps and other things to extract patches
        overlap = round(self.args.input_patch_size * self.args.overlap_porcent)
        overlap -= overlap % 2
        stride = self.args.input_patch_size - overlap
        step_row = (stride - self.img_rows % stride) % stride
        step_col = (stride - self.img_cols % stride) % stride
        k1, k2 = (self.img_rows + step_row)//stride, (self.img_cols + step_col)//stride
        pad_tuple_images = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col), (0 , 0))
        pad_tuple_refere = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col))
        coordinates = np.zeros((k1 * k2, 4))
        counter = 0
        for i in range(k1):
            for j in range(k2):
                
                coordinates[counter, 0] = i * stride
                coordinates[counter, 1] = j * stride
                coordinates[counter, 2] = i * stride + self.args.input_patch_size
                coordinates[counter, 3] = j * stride + self.args.input_patch_size

                counter += 1   
        if self.args.phase == "train":
            self.setmask = self.dataobject.setmasks
            
            self.pastreference = self.dataobject.past_reference
            self.curnreference = self.dataobject.curn_reference
            
            mask_padded = np.pad(self.setmask, pad_tuple_refere, mode = 'symmetric')
            curnref_padded = np.pad(self.curnreference, pad_tuple_refere, mode = 'symmetric')
            bar =  Bar('Processing samples...', max = len(coordinates))
            coordinates_ = coordinates.copy()
            coordinates = []
            for i in range(coordinates_.shape[0]):
                mask_reference_value = mask_padded[int(coordinates_[i , 0]) : int(coordinates_[i , 2]) , int(coordinates_[i , 1]) : int(coordinates_[i , 3])]
                actual_reference_value = curnref_padded[int(coordinates_[i , 0]) : int(coordinates_[i , 2]) , int(coordinates_[i , 1]) : int(coordinates_[i , 3])]
                test_pixels = np.transpose(np.array(np.where(mask_reference_value == 2)))
                # Looking for a test pixels in the mask reference
                if test_pixels.shape[0] == 0:
                    
                    if self.source:
                        number_positives_actual_reference = np.transpose(np.array(np.where(actual_reference_value == 1))).shape[0]
                        porcent_of_positive_pixels_in_actual_reference_i = (number_positives_actual_reference/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                        if porcent_of_positive_pixels_in_actual_reference_i > self.args.porcent_pos_current_ref:
                            set_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == self.set)))
                            porcent_of_set_pixels = (set_pixels_indexs.shape[0]/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                            if porcent_of_set_pixels > 70:
                                coordinates.append(coordinates_[i , :])
                    else:
                        set_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == self.set)))
                        porcent_of_set_pixels = (set_pixels_indexs.shape[0]/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                        if porcent_of_set_pixels > 70:
                            coordinates.append(coordinates_[i , :])

                bar.next()
            bar.finish()
            coordinates = np.array(coordinates)
        
        RESULTS = {'Coordinates':coordinates,
                   'k1': k1, 
                   'k2': k2, 
                   'step_row':step_row, 
                   'step_col':step_col, 
                   'stride':stride, 
                   'overlap':overlap,
                   'pad_tuple_refere': pad_tuple_refere,
                   'pad_tuple_images': pad_tuple_images,
               }
        return RESULTS


class CustomDatasetClassification():
    def __init__(self, dataobject, classes, set, args):
        self.args = args
        self.dataobject = dataobject
        self.classes = classes
        self.img_rows = self.dataobject.image_t1.shape[0]
        self.img_cols = self.dataobject.image_t1.shape[1]
        self.set = set
        self.tar_transform = ToTargetTensor()
        if set == 1 or set == 3:
            self.train_transforms = T.Compose([T.ToTensor(),
                                            T.RandomApply([
                                                T.RandomHorizontalFlip(p=0.5),
                                                T.RandomVerticalFlip(p=0.5),
                                                #T.RandomRotation(degrees=(0, 180)),
                                            ],p=0.8),
                                            ])
            
            coordinates_dict = self.Corner_Coordinates_Definition()
            self.coordinates = coordinates_dict["Coordinates"]
            self.images_padded = np.pad(np.concatenate((self.dataobject.image_t1, self.dataobject.image_t2), axis = 2), 
                                                        coordinates_dict["pad_tuple_images"], mode = 'symmetric')
            self.num_channels = self.images_padded.shape[2]
            self.references_padded = np.pad(self.dataobject.reference, coordinates_dict["pad_tuple_refere"], mode = 'symmetric')
            # Computing labels frequency
            labels_range = np.zeros(1,)
            self.labels_quant = np.zeros((1, self.classes))
            self.labels_freqy = np.zeros((1, self.classes))
            for coordinate in self.coordinates:
                tar = self.references_padded[int(coordinate[0]) : int(coordinate[2]), int(coordinate[1]) : int(coordinate[3])]
                categories = np.unique(tar)
                labels_range = np.concatenate((labels_range, categories), axis = 0)
                for cat in categories:
                        if cat != 255:
                            self.labels_quant[0, int(cat)] += (np.transpose(np.array(np.where(tar == cat)))).shape[0]
                            self.labels_freqy[0, int(cat)] += 1

            self.set_classids = np.unique(labels_range)
            print(np.unique(labels_range))
        else:
            self.img_information = []
            self.img_transform = T.Compose([T.ToTensor()])
            coordinates_dict = self.Corner_Coordinates_Definition()
            self.coordinates = coordinates_dict["Coordinates"]
            self.images_padded = np.pad(np.concatenate((self.dataobject.image_t1, self.dataobject.image_t2), axis = 2), 
                                                        coordinates_dict["pad_tuple_images"], mode = 'symmetric')
            self.images = np.concatenate((self.dataobject.image_t1, self.dataobject.image_t2), axis = 2)
            self.num_channels = self.images_padded.shape[2]
            self.references = self.dataobject.reference
            self.img_information.append({"image_path": self.dataobject.imaget2_path, "label_root": self.dataobject.curnrf_path ,"image_info": coordinates_dict})

    def __len__(self):
        if self.args.phase == 'train':
            return self.coordinates.shape[0]
        elif self.args.phase == 'test':
            return len(self.img_information)
        
    def __getitem__(self, idx):

        if self.args.phase == "train":
            if torch.is_tensor(idx):
                idx = idx.tolist()
            coords = self.coordinates[idx]
            data_padded = np.concatenate((self.images_padded, self.references_padded[:,:,np.newaxis]), axis = 2)
            data_patch = data_padded[int(coords[0]) : int(coords[2]), int(coords[1]) : int(coords[3]), :]
            data_patch = self.train_transforms(data_patch)

            image = data_patch[:-1, :, :]
            target = data_patch[-1, :, :]
            target_1 = target.clone()
            target_2 = target.clone()

            target_1[target_1 == 255] = -1
            mask = target_1 != -1
            mask = mask * 1
            target_1[target_1 == -1] = 0
            target_hot = torch.nn.functional.one_hot(target_1.long(), num_classes = self.classes).permute(2, 0, 1)
            
            return {"images": image.to(torch.float32), "labels": target_2.long(), "mask": mask.unsqueeze(0), "target_hot": target_hot}

        elif self.args.phase == "test":
            if torch.is_tensor(idx):
                idx = idx.tolist()
            image_info = self.img_information[idx]
            
            image = self.images_padded
            image = self.img_transform(image)
            target_1 = self.tar_transform(self.references)
            
            target_1[target_1 == 255] = -1
            mask = target_1 != -1
            mask = mask * 1
            target_1[target_1 == -1] = 0
            
            return {"images": image.to(torch.float32), "images_org": self.images, "labels": target_1.unsqueeze(0), "mask": mask.unsqueeze(0), "image_info": image_info}

    def Corner_Coordinates_Definition(self):
        # Computing the overlaps and other things to extract patches
        overlap = round(self.args.input_patch_size * self.args.overlap_porcent)
        overlap -= overlap % 2
        stride = self.args.input_patch_size - overlap
        step_row = (stride - self.img_rows % stride) % stride
        step_col = (stride - self.img_cols % stride) % stride
        k1, k2 = (self.img_rows + step_row)//stride, (self.img_cols + step_col)//stride
        pad_tuple_images = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col), (0 , 0))
        pad_tuple_refere = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col))
        coordinates = np.zeros((k1 * k2, 4))
        counter = 0
        for i in range(k1):
            for j in range(k2):
                
                coordinates[counter, 0] = i * stride
                coordinates[counter, 1] = j * stride
                coordinates[counter, 2] = i * stride + self.args.input_patch_size
                coordinates[counter, 3] = j * stride + self.args.input_patch_size

                counter += 1   
        if self.args.phase == "train":
            self.setmask = self.dataobject.setmasks
            self.pastreference = self.dataobject.past_reference
            self.curnreference = self.dataobject.curn_reference
            
            mask_padded = np.pad(self.setmask, pad_tuple_refere, mode = 'symmetric')
            curnref_padded = np.pad(self.curnreference, pad_tuple_refere, mode = 'symmetric')
            bar =  Bar('Processing samples...', max = len(coordinates))
            coordinates_ = coordinates.copy()
            coordinates = []
            for i in range(coordinates_.shape[0]):
                mask_reference_value = mask_padded[int(coordinates_[i , 0]) : int(coordinates_[i , 2]) , int(coordinates_[i , 1]) : int(coordinates_[i , 3])]
                actual_reference_value = curnref_padded[int(coordinates_[i , 0]) : int(coordinates_[i , 2]) , int(coordinates_[i , 1]) : int(coordinates_[i , 3])]
                test_pixels = np.transpose(np.array(np.where(mask_reference_value == 2)))
                # Looking for a test pixels in the mask reference
                if test_pixels.shape[0] == 0:
                    if self.args.task == "deforestation_classifier":
                        number_positives_actual_reference = np.transpose(np.array(np.where(actual_reference_value == 1))).shape[0]
                        porcent_of_positive_pixels_in_actual_reference_i = (number_positives_actual_reference/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                        if porcent_of_positive_pixels_in_actual_reference_i > self.args.porcent_pos_current_ref:
                            set_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == self.set)))
                            porcent_of_set_pixels = (set_pixels_indexs.shape[0]/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                            if porcent_of_set_pixels > 70:
                                coordinates.append(coordinates_[i , :])
                    else:
                        set_pixels_indexs = np.transpose(np.array(np.where(mask_reference_value == self.set)))
                        porcent_of_set_pixels = (set_pixels_indexs.shape[0]/(self.args.input_patch_size * self.args.input_patch_size)) * 100
                        if porcent_of_set_pixels > 70:
                            coordinates.append(coordinates_[i , :])

                bar.next()
            bar.finish()
            coordinates = np.array(coordinates)
        
        RESULTS = {'Coordinates':coordinates,
                   'k1': k1, 
                   'k2': k2, 
                   'step_row':step_row, 
                   'step_col':step_col, 
                   'stride':stride, 
                   'overlap':overlap,
                   'pad_tuple_refere': pad_tuple_refere,
                   'pad_tuple_images': pad_tuple_images,
               }
        return RESULTS


class AMAZON_PA():
    def __init__(self, args) -> None:
        self.args = args
        self.area_avoided = 69
        self.path = self.args.data_path + "/AMAZON/"
        self.image1 = "02_08_2016_image_R225_62_PA.npy"
        self.image2 = "20_07_2017_image_R225_62_PA.npy"
        self.imaget1_path = self.path + "IMAGES/" + self.image1
        self.imaget2_path = self.path + "IMAGES/" + self.image2
        
        self.args.horizontal_blocks = 3
        self.args.vertical_blocks = 5
        self.TRAIN_TILES = np.array([1, 7, 9, 13])
        self.VALID_TILES = np.array([5, 12])
        self.UNDESIRED_TILES = np.array([], dtype=np.int32)
        
        self.load_images()
        self.load_deforestation_refers()
        
        self.image_t1, self.image_t2, self.scaler = normalize(self.image_t1, self.image_t2)
        if self.args.phase == 'train' or self.args.phase == 'get_metrics':
            if ssl_mode:
                # SSL pretraining is image-only, so we keep all pixels as train set.
                self.setmasks = np.ones((self.reference.shape[0], self.reference.shape[1]), dtype=np.uint8)
            else:
                self.setmasks = mask_creation(self.reference.shape[0], self.reference.shape[1], 
                                                self.args.horizontal_blocks, self.args.vertical_blocks,
                                                self.TRAIN_TILES, self.VALID_TILES, self.UNDESIRED_TILES)
    def load_images(self):
        print("Loading the images")
        self.image_t1 = np.load(self.imaget1_path)[:,1:1099,:].astype(np.float32)
        self.image_t2 = np.load(self.imaget2_path)[:,1:1099,:].astype(np.float32)
        self.image_t1 = np.transpose(self.image_t1, (1, 2, 0))
        self.image_t2 = np.transpose(self.image_t2, (1, 2, 0))
        print("Original images loaded. Their shapes is: " + str(np.shape(self.image_t1)))
    
    def load_deforestation_refers(self):
        self.pastrf = "PAST_REFERENCE_FROM_1988_2017_EPSG4674_R225_62.npy"
        self.curnrf = "REFERENCE_2017_EPSG4674_R225_62_PA.npy"
        self.pastrf_path  = self.path + "REFERENCES/" + self.pastrf
        self.curnrf_path  = self.path + "REFERENCES/" + self.curnrf
        self.past_reference = np.load(self.pastrf_path)[1:1099,:]
        self.past_reference = skimage.morphology.dilation(self.past_reference, disk(2))
        
        self.curn_reference = np.load(self.curnrf_path)[1:1099,:]
        curn_reference_dilated = skimage.morphology.dilation(self.curn_reference, disk(2))
        buffer_t2_from_dilation = curn_reference_dilated - self.curn_reference
        curn_reference_eroded  = skimage.morphology.erosion(self.curn_reference , disk(2))
        buffer_t2_from_erosion  = self.curn_reference - curn_reference_eroded
        buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
        self.curn_reference = self.curn_reference - buffer_t2_from_erosion
        buffer_t2[buffer_t2 == 1] = 2
        self.curn_reference = self.curn_reference + buffer_t2
        self.curn_reference[self.curn_reference == 2] = 255
        self.curn_reference[self.past_reference == 1] = 255
        self.reference = self.curn_reference.copy()  
    

class AMAZON_RO():
    def __init__(self, args) -> None:
        self.args = args
        self.area_avoided = 69
        self.path = self.args.data_path + "/AMAZON/"
        self.image1 = "18_07_2016_image_R232_67_RO.npy"
        self.image2 = "21_07_2017_image_R232_67_RO.npy"
        self.imaget1_path = self.path + "IMAGES/" + self.image1
        self.imaget2_path = self.path + "IMAGES/" + self.image2
        
        self.args.horizontal_blocks = 10 
        self.args.vertical_blocks = 10
        self.TRAIN_TILES = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
        self.VALID_TILES = np.array([8, 11, 26, 49, 78])
        self.UNDESIRED_TILES = np.array([], dtype=np.int32)
        
    
        
        self.load_images()
        task_name = str(getattr(self.args, "task", "")).lower()
        self.ssl_mode = ("ssl" in task_name) or bool(getattr(self.args, "ssl_only", False))
        if self.ssl_mode:
            print("[AMAZON_RO] SSL mode detected. Skipping REFERENCES loading and using image-only placeholders.")
            self._create_ssl_placeholders()
        else:
            self.load_deforestation_refers()
       
        self.image_t1, self.image_t2, self.scaler = normalize(self.image_t1, self.image_t2)
        if self.args.phase == 'train' or self.args.phase == 'get_metrics':
            if self.ssl_mode:
                # SSL pretraining is image-only; use all pixels as train partition.
                self.setmasks = np.ones((self.reference.shape[0], self.reference.shape[1]), dtype=np.uint8)
            else:
                self.setmasks = mask_creation(self.reference.shape[0], self.reference.shape[1], 
                                                self.args.horizontal_blocks, self.args.vertical_blocks,
                                                self.TRAIN_TILES, self.VALID_TILES, self.UNDESIRED_TILES)
            
    def load_images(self):
        print("Loading the images")
        self.image_t1 = np.load(self.imaget1_path)[:,1:2551,1:5121].astype(np.float32)
        self.image_t2 = np.load(self.imaget2_path)[:,1:2551,1:5121].astype(np.float32)
        self.image_t1 = np.transpose(self.image_t1, (1, 2, 0))
        self.image_t2 = np.transpose(self.image_t2, (1, 2, 0))
        print("Original images loaded. Their shapes is: " + str(np.shape(self.image_t1)))
    def load_deforestation_refers(self):
        self.pastrf = "PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67.npy"
        self.curnrf = "REFERENCE_2017_EPSG32620_R232_67_RO.npy"
        self.pastrf_path  = self.path + "REFERENCES/" + self.pastrf
        self.curnrf_path  = self.path + "REFERENCES/" + self.curnrf
        self.past_reference = np.load(self.pastrf_path)[1:2551,1:5121]
        self.past_reference = skimage.morphology.dilation(self.past_reference, disk(2))
        
        self.curn_reference = np.load(self.curnrf_path)[1:2551,1:5121]
        curn_reference_dilated = skimage.morphology.dilation(self.curn_reference, disk(2))
        buffer_t2_from_dilation = curn_reference_dilated - self.curn_reference
        curn_reference_eroded  = skimage.morphology.erosion(self.curn_reference , disk(2))
        buffer_t2_from_erosion  = self.curn_reference - curn_reference_eroded
        buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
        self.curn_reference = self.curn_reference - buffer_t2_from_erosion
        buffer_t2[buffer_t2 == 1] = 2
        self.curn_reference = self.curn_reference + buffer_t2
        self.curn_reference[self.curn_reference == 2] = 255
        self.curn_reference[self.past_reference == 1] = 255
        self.reference = self.curn_reference.copy()  

    def _create_ssl_placeholders(self):
        h, w = self.image_t1.shape[0], self.image_t1.shape[1]
        self.past_reference = np.zeros((h, w), dtype=np.uint8)
        self.curn_reference = np.zeros((h, w), dtype=np.uint8)
        self.reference = np.zeros((h, w), dtype=np.uint8)
    

class CERRADO_MA():
    def __init__(self, args) -> None:
        self.args = args
        self.area_avoided = 11
        self.path = self.args.data_path + "/CERRADO/"
        self.image1 = "18_08_2017_image_R220_63_MA.npy"
        self.image2 = "21_08_2018_image_R220_63_MA.npy"
        self.imaget1_path = self.path + "IMAGES/" + self.image1
        self.imaget2_path = self.path + "IMAGES/" + self.image2
        
        self.args.horizontal_blocks= 5
        self.args.vertical_blocks = 3
        self.TRAIN_TILES = np.array([1, 5, 12, 13])
        self.VALID_TILES = np.array([6, 7])
        self.UNDESIRED_TILES = [] 
        
        self.load_images()
        
        self.load_deforestation_refers()
        
        self.image_t1, self.image_t2, self.scaler = normalize(self.image_t1, self.image_t2)
        if self.args.phase == 'train' or self.args.phase == 'get_metrics':
            self.setmasks = mask_creation(self.reference.shape[0], self.reference.shape[1], 
                                              self.args.horizontal_blocks, self.args.vertical_blocks,
                                              self.TRAIN_TILES, self.VALID_TILES, self.UNDESIRED_TILES)
            
    def load_images(self):
        print("Loading the images")
        self.image_t1 = np.load(self.imaget1_path)[:,:1700,:1440].astype(np.float32)
        self.image_t2 = np.load(self.imaget2_path)[:,:1700,:1440].astype(np.float32)
        self.image_t1 = np.transpose(self.image_t1, (1, 2, 0))
        self.image_t2 = np.transpose(self.image_t2, (1, 2, 0))
        print("Original images loaded. Their shapes is: " + str(np.shape(self.image_t1)))
    def load_deforestation_refers(self):
        self.pastrf = "PAST_REFERENCE_FOR_2018_EPSG4674_R220_63_MA.npy"
        self.curnrf = "REFERENCE_2018_EPSG4674_R220_63_MA.npy"
        self.pastrf_path  = self.path + "REFERENCES/" + self.pastrf
        self.curnrf_path  = self.path + "REFERENCES/" + self.curnrf
        self.past_reference = np.load(self.pastrf_path)[:1700,:1440]
        self.past_reference = skimage.morphology.dilation(self.past_reference, disk(2))
        
        self.curn_reference = np.load(self.curnrf_path)[:1700,:1440]
        curn_reference_dilated = skimage.morphology.dilation(self.curn_reference, disk(2))
        buffer_t2_from_dilation = curn_reference_dilated - self.curn_reference
        curn_reference_eroded  = skimage.morphology.erosion(self.curn_reference , disk(2))
        buffer_t2_from_erosion  = self.curn_reference - curn_reference_eroded
        buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
        self.curn_reference = self.curn_reference - buffer_t2_from_erosion
        buffer_t2[buffer_t2 == 1] = 2
        self.curn_reference = self.curn_reference + buffer_t2
        self.curn_reference[self.curn_reference == 2] = 255
        self.curn_reference[self.past_reference == 1] = 255
        self.reference = self.curn_reference.copy()  
    
    
