from snn_conv import *
import torchvision
from torchvision import transforms
import torch.optim as optim
from utils import *
from mnist_data import load_mnist
import numpy as np
import os

train_loss_list = []
test_acc_list = []

num_steps = 10
batch_size = 64
lr = 0.1
leak_mem = 0.99
num_epochs = 50
img_size = 28
num_classes = 10

model = SNN_Conv(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_classes=num_classes)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
best_acc = 0

mnist_train, mnist_test = load_mnist(batch_size=batch_size, subset=num_classes)

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
