import argparse
import os
import json
import shutil
import numpy as np
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter
import pickle

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models

from better_mistakes.util.rand import make_deterministic
from better_mistakes.util.folders import get_expm_folder
from better_mistakes.util.label_embeddings import create_embedding_layer
from better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from better_mistakes.util.config import load_config
from better_mistakes.data.softmax_cascade import SoftmaxCascade
from better_mistakes.data.transforms import train_transforms, val_transforms
from better_mistakes.model.init import init_model_on_gpu
from better_mistakes.model.run_xent import run
from better_mistakes.model.run_nn import run_nn
from better_mistakes.model.labels import make_all_soft_labels
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes

from helper import *
import math
import pickle
import heapq

torch.backends.cudnn.benchmark = True
MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
LOSS_NAMES = ["cross-entropy", "soft-labels", "hierarchical-cross-entropy", "cosine-distance", "ranking-loss", "cosine-plus-xent", "yolo-v2"]
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD"]
DATASET_NAMES = ["tiered-imagenet-84", "inaturalist19-84", "tiered-imagenet-224", "inaturalist19-224"]



def _load_checkpoint(opts, model, optimizer,model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,map_location='cuda:0')
        opts.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps = checkpoint["steps"]
        print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint["epoch"]))
    elif opts.pretrained_folder is not None:
        if os.path.exists(opts.pretrained_folder):
            print("=> loading pretrained checkpoint '{}'".format(opts.pretrained_folder))
            if os.path.isdir(opts.pretrained_folder):
                checkpoint = torch.load(os.path.join(opts.pretrained_folder, "checkpoint.pth.tar"))
            else:
                checkpoint = torch.load(opts.pretrained_folder)
            if opts.devise or opts.barzdenzler:
                model_dict = model.state_dict()
                pretrained_dict = checkpoint["state_dict"]
                # filter out FC layer
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ["fc.1.weight", "fc.1.bias"]}
                # overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # load the new state dict
                model.load_state_dict(pretrained_dict, strict=False)
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            steps = 0
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_folder, checkpoint["epoch"]))
        else:
            raise FileNotFoundError("Can not find {}".format(opts.pretrained_folder))
    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps
def _select_optimizer(model, opts):
    if opts.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == "adam_amsgrad":
        if opts.devise or opts.barzdenzler:
            return torch.optim.Adam(
                [
                    {"params": model.conv1.parameters()},
                    {"params": model.layer1.parameters()},
                    {"params": model.layer2.parameters()},
                    {"params": model.layer3.parameters()},
                    {"params": model.layer4.parameters()},
                    {"params": model.fc.parameters(), "lr": opts.lr_fc, "weight_decay": opts.weight_decay_fc},
                ],
                lr=opts.lr,
                weight_decay=opts.weight_decay,
                amsgrad=True,
            )
        else:
            return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False, )
    else:
        raise ValueError("Unknown optimizer", opts.loss)

def softmax(x):
    '''Compute softmax values for a single vector.'''
    return np.exp(x) / np.sum(np.exp(x))

def row_softmax(output):
    '''Compute Row-Wise SoftMax given a matrix of logits'''
    new=np.array([softmax(i) for i in output])
    return new

def topk_accuracy(prediction,target,k=5):
    ind=heapq.nlargest(k, range(len(prediction)), prediction.take)
    for i in ind:
        if i==target:
            return 1
    return 0



def get_all_cost_sensitive(output,distances,classes):
    '''Re-Rank all predictions in the dataset using CRM'''
    
    num_classes=len(classes)
    C=[[0 for i in range(num_classes)] for j in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j]=distances[(classes[i],classes[j])]

    final=np.dot(output,C)
    return -1*final

    

def get_topk(prediction,target,distances,classes,k=1):
    '''Computing hierarchical distance@k'''
    ind=heapq.nlargest(k, range(len(prediction)), prediction.take)
    scores=[]
    s1,s2=0,0
    for i in ind:
        scores.append(distances[(classes[i],classes[target])])
    return scores



def get_metrics(opts,output,target,distances,classes):


    ##Random Shuffling if Required
    if opts.shuffle_classes==1:
        np.random.seed(42)##Seed used to Train HXE/Soft-Labels. However, can be changed
        np.random.shuffle(classes)

    ##Apply CRM
    if opts.rerank==1:
        output=get_all_cost_sensitive(output,distances,classes)

    orig_top1=[]
    orig_mistake=[]
    orig_avg_1=[]
    orig_avg_5=[]
    orig_avg_20=[]



    for i in range(len(output)):

        if output[i].argmax()==target[i]:
            orig_top1.append(1)
        else:
            orig_top1.append(0)
            orig_mistake.append(distances[(classes[target[i]],classes[output[i].argmax()])])

        orig_avg_1.extend(get_topk(output[i],target[i],distances,classes,1))

        orig_avg_5.append(get_topk(output[i],target[i],distances,classes,5))

        orig_avg_20.append(get_topk(output[i],target[i],distances,classes,20))


    print("Top-1 Accuracy",np.array(orig_top1).mean())
    
    print("Mistake Severity",np.array(orig_mistake).mean())

    print("Hierarchical Distance@1",np.array(orig_avg_1).mean())

    print("Hierarchical Distance@5",np.array(orig_avg_5).mean())

    print("Hierarchical Distance@20",np.array(orig_avg_20).mean())
    result=[np.array(orig_top1).mean(),np.array(orig_avg_1).mean(),np.array(orig_avg_5).mean(),np.array(orig_avg_20).mean(),
    np.array(orig_mistake).mean()]
    
    return result    

 

def main(opts,model_path):


    test_dir = os.path.join(opts.data_path, "test")
    test_dataset = datasets.ImageFolder(test_dir, val_transforms(opts.data, resize=(224,224),normalize=True))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers, pin_memory=True, drop_last=False)
    gpus_per_node=1
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data, opts.data_dir)
    if opts.loss == "yolo-v2":
        classes, _ = get_classes(hierarchy, output_all_nodes=True)
    else:
        classes = test_dataset.classes
    opts.num_classes = len(classes)


    if opts.loss == "yolo-v2":

        cascade = SoftmaxCascade(hierarchy, classes).cuda(opts.gpu)
        num_leaf_classes = len(hierarchy.treepositions("leaves"))
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)

        def yolo2_corrector(output):
            return cascade.final_probabilities(output)[:, :num_leaf_classes]



    model = init_model_on_gpu(gpus_per_node, opts)

    # setup optimizer
    optimizer = _select_optimizer(model, opts)

    # load from checkpoint if existing
    steps = _load_checkpoint(opts, model, optimizer,model_path)

    corrector = yolo2_corrector if opts.loss == "yolo-v2" else lambda x: x

    model.eval()
    torch.no_grad()


    test_output=[]
    test_target=[]
    for batch_idx,(embeddings,target) in enumerate(test_loader):
        if opts.gpu is not None:
            embeddings = embeddings.cuda(opts.gpu,non_blocking=True)

        output=model(embeddings)
        output=corrector(output)
        
        test_output.extend(output.cpu().tolist())
        test_target.extend(target.tolist())

    test_output=np.array(test_output)
    test_target=np.array(test_target)

    if opts.loss!='yolo-v2':
        softmax_output=row_softmax(test_output)
    else:
        softmax_output=test_output
    model_ece=guo_ECE(softmax_output,test_target)
    model_mce=MCE(softmax_output,test_target)
    print("ECE:",model_ece)
    print("MCE:",model_mce)
    result=get_metrics(opts,softmax_output,test_target,distances,classes)
    result.append(model_ece)
    result.append(model_mce)
    return result




if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None, help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--num_training_steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    parser.add_argument("--shuffle_classes", default=False, type=boolean, help="Shuffle classes in the hierarchy")
    parser.add_argument("--beta", default=0, type=float, help="Softness parameter: the higher, the closer to one-hot encoding")
    parser.add_argument("--alpha", type=float, default=0, help="Decay parameter for hierarchical cross entropy.")
    # Devise/B&D ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--devise", type=boolean, default=False, help="Use DeViSe label embeddings")
    parser.add_argument("--devise_single_negative", type=boolean, default=False, help="Use one negative per samples instead of all")
    parser.add_argument("--barzdenzler", type=boolean, default=False, help="Use Barz&Denzler label embeddings")
    parser.add_argument("--train_backbone_after", default=float("inf"), type=float, help="Start training backbone too after this many steps")
    parser.add_argument("--use_2fc", default=False, type=boolean, help="Use two FC layers for Devise")
    parser.add_argument("--fc_inner_dim", default=1024, type=int, help="If use_2fc is True, their inner dimension.")
    parser.add_argument("--lr_fc", default=1e-3, type=float, help="learning rate for FC layers")
    parser.add_argument("--weight_decay_fc", default=0.0, type=float, help="weight decay of FC layers")
    parser.add_argument("--use_fc_batchnorm", default=False, type=boolean, help="Batchnorm layer in network head")
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="tiered-imagenet-224", help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int, help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="../data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="../data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default=None, help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str, help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=5, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--out_folder",default=None,type=str,help="Path to model checkpoint")

    ## CRM ----------------------------------------------------------------------------------
    parser.add_argument("--rerank",default=0,type=int,help='whether to use CRM or not')


    ### Logs ---------------------------------------------------------------------------------
    parser.add_argument("--expname",default='cross-entropy',type=str,help="Name of model")
    parser.add_argument("--epoch1",default=10,type=int,help="First epoch to evaluate")
    parser.add_argument("--out_folder1",default=None,type=str,help="Path to model checkpoint")
    parser.add_argument("--out_folder2",default=None,type=str,help="Path to model checkpoint")
    parser.add_argument("--out_folder3",default=None,type=str,help="Path to model checkpoint")
    parser.add_argument("--out_folder4",default=None,type=str,help="Path to model checkpoint")
    parser.add_argument("--out_folder5",default=None,type=str,help="Path to model checkpoint")


    opts=parser.parse_args()
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    logs=[]
    ##Evaluating Results on 5 checkpoints
    logs.append(main(opts,opts.out_folder1))
    logs.append(main(opts,opts.out_folder2))
    logs.append(main(opts,opts.out_folder3))
    logs.append(main(opts,opts.out_folder4))
    logs.append(main(opts,opts.out_folder5))
    logs=np.array(logs,dtype='float64')
    savename=opts.expname
    np.savetxt(savename,logs, fmt="%.5f", delimiter=",")


