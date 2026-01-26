import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from progress.bar import Bar
import matplotlib.pyplot as plt
import skimage.morphology
from options.visualoptions import VisualOptions
from options.testoptions import TestOptions
from data.DeforestationDataset import *

def Area_under_the_curve(X, Y):

    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])

    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b                
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))
                    
    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))
    # plt.figure(3); plt.stem(X, Y)
    # plt.figure(4); plt.stem(X_, Y_)
    # sys.exit()
    
    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)
    
    return area
def correct_nan_values(arr, before_value, last_value):
    
    before = np.zeros_like(arr)
    after = np.zeros_like(arr)
    arr_ = arr.copy()
    index = 0   
    if before_value == 1: before_value = arr[~np.isnan(arr)][0]
    if last_value == 1: last_value = arr[~np.isnan(arr)][-1]

    for i in range(len(arr)):
        before[i] = before_value            
        if not np.isnan(arr[i]):
            if i: after[index:i] = arr[i] 
            before_value = arr[i]                
            index = i
    after[index:len(arr)] = last_value
    
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr_[i] = (before[i] + after[i]) / 2
    
    return arr_

def get_class_labels(dataset_name):
    if dataset_name == "deforestation":
        return ["Forest", "Deforestation"]
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))

def plot_precision_recall_curve(precision, recall, ap, output_folder, prefix):
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'AP: {ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{prefix}_Precision_Recall.png'))
    plt.close()

def apply_area_criteria(predictions, mask, area_threshold):
    # Reshape predictions to match mask shape if needed
    predictions = predictions.reshape(mask.shape)

    positive_map_init_ = skimage.morphology.area_opening(predictions.astype(int), area_threshold=area_threshold, connectivity=1)
    eliminated_samples = (predictions - positive_map_init_) - 1
    eliminated_samples[eliminated_samples == -1] = 1

    eliminated_samples = eliminated_samples.reshape(mask.shape)
    final_mask = mask * eliminated_samples
    return final_mask

def compute_metrics_for_folder(folder_path, set_classesids, test_mask, dataset):
    results_files = os.listdir(folder_path)
    precisions, recalls, thresholds = [], [], []
    cm_total = np.zeros((len(set_classesids), len(set_classesids)))
    print("Computing metrics for folder: ", folder_path)
    for reference in results_files:
        predictionfile_path = os.path.join(folder_path, reference)
        if os.path.isfile(predictionfile_path):
            predictions = np.load(predictionfile_path)
            predic_label = predictions[:,:,14]
            ground_truth = predictions[:, :, 15].reshape(-1)
            probabilities = predictions[:, :, -1].flatten()
            mask_ = predictions[:, :, 16]

            # Determine thresholds
            unique_probs = np.unique(probabilities)
            thresholds = np.sort(np.linspace(unique_probs.min(), unique_probs.max(), num=100))[::-1]
            # Compute precision-recall curve
            # Bar indicating the progress of the loop
            barth = Bar('Processing', max=len(thresholds))
            for threshold in thresholds:
                
                predic_label_ = (probabilities >= threshold).astype(int)
                predic_label_ = predic_label_.reshape(mask_.shape)  # Ensure shape consistency

                # Apply area threshold criteria
                final_mask = apply_area_criteria(predic_label_, mask_ * test_mask, dataset.area_avoided)
                indexs = np.where(final_mask.flatten() == 1)

                if len(indexs[0]) == 0:
                    continue

                T = ground_truth[indexs]
                P = predic_label_.flatten()[indexs]

                # Collect precision and recall
                if np.sum(T) > 0:
                    cm_total = confusion_matrix(T, P, labels=set_classesids)
                    TP = cm_total[1, 1]
                    FP = cm_total[0, 1]
                    FN = cm_total[1, 0]
                    TN = cm_total[0, 0]

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
                    precisions.append(precision)
                    recalls.append(recall)
                barth.suffix = f'Threshold: {threshold:.1f} | Precision: {precision:.2f} | Recall: {recall:.2f}'
                barth.next()
            barth.finish()
    
    # Average Precision
    if len(recalls) > 0 and len(precisions) > 0:
        precisions_ = correct_nan_values(np.array(precisions), 1, 0)
        recalls_ = correct_nan_values(np.array(recalls), 0, 1)
    
        # Correction of depresions in the curve       
        precision_depr = np.maximum.accumulate(precisions_[::-1])[::-1]

        if recalls_[0] > 0:
            recalls_ = np.concatenate((np.array([0]), recalls_), axis=0)
            precisions_ = np.concatenate((precisions_[0:1], precisions_), axis=0)
            precision_depr = np.concatenate((precision_depr[0:1], precision_depr), axis=0)
    
        ap = Area_under_the_curve(recalls_, precision_depr)
    else:
        ap = 0

    # Compute overall metrics for constant threshold (e.g., 0.5)
    final_mask = apply_area_criteria(predic_label, mask_ * test_mask, dataset.area_avoided)
    indexs = np.where(final_mask.flatten() == 1)
    T = ground_truth[indexs]
    P = predic_label.flatten()[indexs]

    cm_total = confusion_matrix(T, P, labels=set_classesids)
    TP = cm_total[1, 1]
    FP = cm_total[0, 1]
    FN = cm_total[1, 0]
    TN = cm_total[0, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / cm_total.sum()

    return accuracy, precision, recall, f1_score, ap, final_mask, np.array(precisions), np.array(recalls)

def main():
    args = TestOptions().initialize()
    dataset = DeforestationDataset(args)
    test_mask = dataset.dataobject.setmasks
    test_mask[test_mask != 2] = 0
    test_mask[test_mask == 2] = 1
    classes = get_class_labels(args.dataset_name)
    class_ids = [0, 1]
    args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
    print(args.results_savepath)
    training_domain = args.experiment_name.split('_')[1]
    
    if not os.path.exists(args.results_savepath):
        print("The current folder: " + args.results_savepath + "doesn't exists")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    
    results_folders = os.listdir(args.results_savepath)
    print(results_folders)
    if len(results_folders) == 0:
        print("The current folder: " + args.results_folders + "doesn't contains evaluated moodels")
        print("Please, make sure you are addressing the right results folders")
        sys.exit()

    ACCURACIES, PRECISIONS, RECALLS, F1_SCORES, APS = [], [], [], [], []

    for result_folder in results_folders:
        # Check if the foder is a valid result folder
        if not os.path.isdir(os.path.join(args.results_savepath, result_folder)):
            continue
        if "domains" in args:
            folder_path = os.path.join(args.results_savepath, result_folder, 'predictions_' + args.eval_type + "_" + args.domains)
        else:
            folder_path = os.path.join(args.results_savepath, result_folder, 'predictions_' + args.eval_type)

        accuracy, precision, recall, f1_score, ap, final_mask, pr_precisions, pr_recalls = compute_metrics_for_folder(
            folder_path, class_ids, test_mask, dataset)
        
        # Store metrics
        ACCURACIES.append(accuracy)
        PRECISIONS.append(precision)
        RECALLS.append(recall)
        F1_SCORES.append(f1_score)
        APS.append(ap)

        # Save individual precision-recall curve
        visual_folder = os.path.join(args.results_savepath, result_folder, 'metrics_' + args.eval_type + "_" + args.domains)
       
        if not os.path.exists(visual_folder):
            os.makedirs(visual_folder)
        plot_precision_recall_curve(pr_precisions, pr_recalls, ap, visual_folder, result_folder)

        # Save individual metrics
        ind_results_path = os.path.join(visual_folder, 'Individual_Results.csv')
        pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Average Precision'],
            'Score': [accuracy, precision, recall, f1_score, ap]
        }).to_csv(ind_results_path, index=False)
        cv2.imwrite(visual_folder + "/test_mask.png", 255 * final_mask)

    # Compute averages over all folders
    avg_metrics = {
        'Accuracy': [np.mean(ACCURACIES), np.std(ACCURACIES), np.min(ACCURACIES), np.max(ACCURACIES), np.median(ACCURACIES)], 
        'Precision': [np.mean(PRECISIONS), np.std(PRECISIONS), np.min(PRECISIONS), np.max(PRECISIONS), np.median(PRECISIONS)], 
        'Recall': [np.mean(RECALLS), np.std(RECALLS), np.min(RECALLS), np.max(RECALLS), np.median(RECALLS)], 
        'F1 Score': [np.mean(F1_SCORES), np.std(F1_SCORES), np.min(F1_SCORES), np.max(F1_SCORES), np.median(F1_SCORES)],
        'Average Precision': [np.mean(APS), np.std(APS), np.min(APS), np.max(APS), np.median(APS)]
    }

    # Save average metrics
    avg_results_path = os.path.join(args.results_savepath, args.domains + '_Average_Results.csv')
    pd.DataFrame(avg_metrics).to_csv(avg_results_path, index=False)

if __name__ == '__main__':
    main()