import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics


class AUCRecorder(object):
    def __init__(self):
        self.prediction = []
        self.target = []

    def update(self,prediction,target):
        self.prediction = self.prediction + prediction.tolist()
        self.target = self.target + target.tolist()

    def auc(self):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def draw_roc(self, path):
        prediction = np.array(self.prediction)
        target = np.array(self.target)
        fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(4.5,4.5))

        x = np.arange(0,1.01,0.01)

        plt.plot(x,x,ls="--",color='grey',alpha=0.5)
        plt.plot(fpr,tpr,label='auc {:.3f}'.format(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(path,dpi=300)



class evaluate(object):
    def __init__(self):
        self.auc_record=AUCRecorder()
        self.accuracy_record=[]
        self.total_len=0
        self.loss_record=[]
        self.best_auc=[-1,-1]
        self.epochs=[]
        self.n_epoch=0
        self.true_pos=0
        self.true_neg=0
        self.false_pos=0
        self.false_neg=0

    def validate(self,predicted,label):
        p=torch.argmax(predicted,dim=1)
        correct=0
        true_pos, true_neg, false_pos, false_neg =0,0,0,0
        for i, m in enumerate(p):
            if m == label[i]:
                correct += 1
                if label[i]==1:
                    true_pos+=1
                else:
                    true_neg+=1
            else:
                if label[i]==1:
                    false_neg+=1
                else:
                    false_pos+=1
        return correct, true_pos, true_neg, false_pos, false_neg

    def update(self,predicted,label,loss):
        self.total_len+=len(label)
        self.auc_record.update(predicted[:,1],label)
        correct, true_pos, true_neg, false_pos, false_neg=self.validate(predicted, label)
        self.true_neg += true_neg
        self.true_pos += true_pos
        self.false_neg += false_neg
        self.false_pos += false_pos
        self.accuracy_record=np.append(self.accuracy_record,correct)
        self.loss_record=np.append(self.loss_record,loss)

    def total_acc(self):
        return np.sum(self.accuracy_record)/self.total_len

    def confusion_matrix(self):
        return [[self.true_pos,self.false_pos],[self.false_neg,self.true_neg]]

    def iter_acc(self):
        return self.accuracy_record

    def draw_roc(self,path):
        self.auc_record.draw_roc(path)

    def avg_loss(self):
        return np.sum(self.loss_record)/len(self.loss_record)

    def iter_loss(self):
        return self.loss_record

    def total_loss(self):
        return np.sum(self.loss_record)

    def auc_score(self):
        return self.auc_record.auc()

    def new_epoch(self):
        if self.best_auc[0]<self.auc_score():
            self.best_auc=[self.auc_score(),self.n_epoch]
        self.epochs=np.append(self.epochs,[{"auc":self.auc_record,"acc":self.loss_record,"loss":self.loss_record}])
        self.loss_record=[]
        self.accuracy_record=[]
        self.auc_record=AUCRecorder()
        self.total_len=0
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.n_epoch+=1

    def set_best_auc(self,val):
        self.best_auc[0]=val
        self.best_auc[1]=-1

    def find_best_auc(self):
        aux=None
        index=self.best_auc[1]
        if index!=-1:
            aux=self.epochs[index]
        return index,self.best_auc[0],aux


