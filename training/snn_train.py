from models.snn_conv import SNN_Conv
from torch import nn, optim
from utils.utils import *
from training_data.mnist_data import load_mnist
import numpy as np
import pandas as pd
import os

# computes accuracy and flops as model trains, for each epoch in the passed list

def run(batch_size=64, num_steps=10, lr=0.1, leak_mem=0.99, num_epochs=50, epoch_list=None, img_size=28, num_classes=10, case_study='accuracy', df=None):

    # initialize flops computation
    save_flops = False if case_study == 'accuracy' else True

    #initialize model
    model = SNN_Conv(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_classes=num_classes, save_flops=save_flops)
    model = model.cuda()

    # set optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)

    # load mnist data
    mnist_train, mnist_test = load_mnist(batch_size=batch_size, subset=10)

    # loop through epochs, run forward and backward pass.
    # adjust optimizer and learning rate
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        model.train()
        for i, data in enumerate(mnist_train):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            output = model(inputs)

            loss   = criterion(output, labels)

            prec1, prec5 = accuracy(output, labels, topk=(1, 5))
            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()

        
        print("Epoch: {}/{};".format(epoch+1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

        adjust_learning_rate(optimizer, epoch, num_epochs)

        # evaluatate model for accuracy
        if (epoch+1) in epoch_list and case_study != 'flops':
            acc_top1, acc_top5 = [], []
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(mnist_test, 0):

                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()

                    out = model(images)
                    prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                    acc_top1.append(float(prec1))
                    acc_top5.append(float(prec5))


            test_accuracy = np.mean(acc_top1)
            print ("test_accuracy : {}". format(test_accuracy))

        # save run metrics
        if epoch+1 in epoch_list and case_study != 'flops':
            if case_study == 'accuracy':
                new_df = pd.DataFrame([[batch_size, lr, leak_mem, num_steps, epoch+1, train_loss.avg, format(test_accuracy)]], columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy'])
            else:
                new_df = pd.DataFrame([[batch_size, lr, leak_mem, num_steps, epoch+1, train_loss.avg, format(test_accuracy), model.total_flops]], columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy', 'flops'])
            df = pd.concat([df, new_df], ignore_index=True)

    if case_study == 'flops':
        df = pd.DataFrame([[batch_size, lr, leak_mem, num_steps, epoch+1, model.total_flops]], columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'flops'])

    return df