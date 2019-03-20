"""Metric file for pixel wise scores computation
"""

import argparse
import math
import numpy as np
from PIL import Image
import os

import sklearn.metrics as metrics # for confusion matrix

#==============================================
#==============================================
# IO FUNCTIONS
#==============================================
#==============================================

def raster_loader(file_path):
    """ Load a raster of .
    """
    image = np.array(Image.open(file_path), dtype=np.int)
    return image

#==============================================
#==============================================
# STATS
#==============================================
#==============================================

def stats_overall_accuracy(cm):
    """Compute the overall accuracy.
    """
    return np.trace(cm)/cm.sum()

def stats_pfa_per_class(cm):
    """Compute the probability of false alarms.
    """
    sums = np.sum(cm, axis=0) 
    mask = (sums>0)
    sums[sums==0] = 1
    pfa_per_class = (cm.sum(axis=0)-np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class

def stats_accuracy_per_class(cm):
    """Compute the accuracy per class and average
        puts -1 for invalid values (division per 0)
        returns average accuracy, accuracy per class
    """
    # equvalent to for class i to 
    # number or true positive of class i (data[target==i]==i).sum()/ number of elements of i (target==i).sum()
    sums = np.sum(cm, axis=1) 
    mask = (sums>0)
    sums[sums==0] = 1
    accuracy_per_class = np.diag(cm) / sums #sum over lines
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy, accuracy_per_class

def stats_iou_per_class(cm):
    """Compute the iou per class and average iou
        Puts -1 for invalid values
        returns average iou, iou per class
    """
    
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mask  = (sums>0)
    sums[sums==0] = 1
    iou_per_class = np.diag(cm) / sums
    iou_per_class[np.logical_not(mask)] = -1
    average_iou = iou_per_class[mask].mean()
    return average_iou, iou_per_class

def stats_f1score_per_class(cm):
    """Compute f1 scores per class and mean f1.
        puts -1 for invalid classes
        returns average f1 score, f1 score per class
    """
    # defined as 2 * recall * prec / recall + prec
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0))
    mask  = (sums>0)
    sums[sums==0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score =  f1score_per_class[mask].mean()
    return average_f1_score, f1score_per_class


def main():
    """Main function."""

    # create the parser for arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="input file")
    parser.add_argument("--target", type=str, default=None,
                        help="target file (ground truth)")
    parser.add_argument("--filelist", type=str, default=None,
                        help="filepath for multi-file stats")
    parser.add_argument("--labels", type=int, default=None, required=True,
                        help="number of labels, if not used, computed from data")
    parser.add_argument("--delimiter", type=str, default=" ", 
                        help="Default delimiter for loadin file, default is space")
    parser.add_argument("--verbose", action="store_true",
                        help="Detailed per class values")
    args = parser.parse_args()

    print("LABELS", args.labels)

    labels = list(range(args.labels))
    cm = None # the confusion matrix

    print("Loading data and build confusion matrix...", flush=True)
    if args.filelist is not None:
        in_f = open(args.filelist, "r")
        for line in in_f:
            filenames = line.split("\n")[0].split(args.delimiter)
            print("  ", filenames[0])
            data = raster_loader(filenames[0])
            target = raster_loader(filenames[1])
            data = data.ravel()
            target = target.ravel()
            if cm is None:
                cm = metrics.confusion_matrix(target,data, labels=labels)
            else:
                cm += metrics.confusion_matrix(target,data, labels=labels)
        in_f.close()
    else:
        if args.input is None or args.target is None:
            raise Exception("Input / Target exception")
        data = raster_loader(args.input)
        target = raster_loader(args.target)

        data = data.ravel()
        target = target.ravel()
        cm = metrics.confusion_matrix(target,data, labels=labels)
    print("Done")

    if args.verbose:
        print("============ Confusion Matrix")
        print(cm)

    overall_accuracy = stats_overall_accuracy(cm)
    print("Overall Accuracy", overall_accuracy)


    average_accuracy, accuracy_per_class = stats_accuracy_per_class(cm)
    if args.verbose:
        print("============ Accuracy per class")
        for i in range(args.labels):
            if accuracy_per_class[i] > -1:
                print("  label", i, accuracy_per_class[i])
            else:
                print("  label", i, "invalud value (not in ground truth)")
    print("Average accuracy", average_accuracy)

    average_iou, iou_per_class = stats_iou_per_class(cm)
    if args.verbose:
        print("============ Intersection over union")
        for i in range(args.labels):
            if iou_per_class[i] > -1:
                print("  label", i, iou_per_class[i])
            else:
                print("  label", i, "invalud value (not in ground truth)")
    print("Average IoU", average_iou)

    average_f1_score, f1score_per_class = stats_f1score_per_class(cm)
    if args.verbose:
        print("============ F1-scores")
        for i in range(args.labels):
            if f1score_per_class[i] > -1:
                print("  label", i, f1score_per_class[i])
            else:
                print("  label", i, "invalud value (not in ground truth)")
    print("Average F1-score", average_f1_score)

    average_pfa, pfa_per_class = stats_pfa_per_class(cm)
    if args.verbose:
        print("============ PFA per class")
        for i in range(args.labels):
            if pfa_per_class[i] > -1:
                print("  label", i, pfa_per_class[i])
            else:
                print("  label", i, "invalud value (not in ground truth)")
    print("Average PFA", average_pfa)




#==============================================
if __name__ == "__main__":
    main()

#EOF
