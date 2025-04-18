from snn_conv import *
import torchvision
from torchvision import transforms
import torch.optim as optim
from utils import *
from mnist_data import load_mnist
import numpy as np
import os
import pandas as pd

def snn_run(batch_size=64, num_steps=10, lr=0.1, leak_mem=0.99, num_epochs=50, img_size=28, num_classes=10):

    train_loss_list = []
    test_acc_list = []

    path = f'snn/{batch_size}/{num_steps}/{num_epochs}/{leak_mem}'
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = f'{path}/flops.txt'

    with open(file_path, 'w') as f:
        f.write("FLOPS\n")
        f.write(f"Conv Neuron Flops: {0}\n")
        f.write(f"Conv Flops: {0}\n")
        f.write(f"Conv BN Flops: {0}\n")
        f.write(f"Pool Flops: {0}\n")
        f.write(f"FC Neuron Flops: {0}\n")
        f.write(f"FC Flops: {0}\n")
        f.write(f"FC BN Flops: {0}\n")
        f.write(f"Total Flops: {0}\n")
    f.close()

    model = SNN_Conv(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_classes=num_classes, path=file_path)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    best_acc = 0

    mnist_train, mnist_test = load_mnist(batch_size=batch_size, subset=10)

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

        if (epoch+1) %  5 == 0:
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


            # Model save
            if best_acc < test_accuracy:
                best_acc = test_accuracy

                model_dict = {
                        'global_step': epoch + 1,
                        'state_dict': model.state_dict(),
                        'accuracy': test_accuracy}

                if not os.path.exists('log/'):
                    os.makedirs('log/')

                torch.save(model_dict, 'log/bestmodel.pth.tar')

    total_flops = model.flops_data['cnn_neuron_flops'] + model.flops_data['cnn_flops'] + model.flops_data['cnn_bn_flops'] + model.flops_data['pool_flops'] + model.flops_data['fc_neuron_flops'] + model.flops_data['fc_flops'] + model.flops_data['fc_bn_flops']

    with open(file_path, 'w') as f:
        f.write("SNN FLOPS\n")
        f.write(f"Conv Neuron Flops: {model.flops_data['cnn_neuron_flops']}\n")
        f.write(f"Conv Flops: {model.flops_data['cnn_flops']}\n")
        f.write(f"Conv BN Flops: {model.flops_data['cnn_bn_flops']}\n")
        f.write(f"Pool Flops: {model.flops_data['pool_flops']}\n")
        f.write(f"FC Neuron Flops: {model.flops_data['fc_neuron_flops']}\n")
        f.write(f"FC Flops: {model.flops_data['fc_flops']}\n")
        f.write(f"FC BN Flops: {model.flops_data['fc_bn_flops']}\n")
        f.write(f"Total Flops: {total_flops}\n")

    return pd.DataFrame([[batch_size, lr, leak_mem, num_steps, num_epochs, train_loss.avg, format(test_accuracy)]], columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_acc'])

def case_study():

    batch_size_list = [8, 16, 32, 64] # 8
    lr_list = [0.2, 0.1, 0.05, 0.01] # 0.2
    num_epochs_list = [5, 10, 25, 50] # 50
    num_steps_list = [1, 5, 10, 25, 50] # 10
    leak_mem_list = [0.995, 0.99, 0.985, 0.98, 0.975] # 0.995

    img_size = 28
    num_classes = 10

    df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_acc'])
    for batch_size in batch_size_list:
        for lr in lr_list:
            for num_epochs in num_epochs_list:
                for num_steps in num_steps_list:
                    for leak_mem in leak_mem_list:
                        print(f"Running with batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, num_steps={num_steps}, leak_mem={leak_mem}")
                        new_df = snn_run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, img_size=img_size, num_classes=num_classes, num_steps=num_steps, leak_mem=leak_mem)

                        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv('snn_results.csv', index=False)

def single_run():
    batch_size = 64
    lr = 0.1
    num_epochs = 50
    leak_mem = 0.99
    num_steps = 10

    img_size = 28
    num_classes = 10

    snn_run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, img_size=img_size, num_classes=num_classes, num_steps=num_steps, leak_mem=leak_mem)