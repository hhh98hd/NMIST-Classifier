import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.utils.data.dataloader
import time
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive')

# Set image size for scaling
IMAGE_SIZE = 227


class AlexNet(nn.Module):
    def __init__(self, drop):
        super(AlexNet, self).__init__()

        # Dropout
        self.drop = nn.Dropout(p=drop)

        # 1st Convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=96,
                              kernel_size=11,
                              stride=4).cuda()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 2nd Convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=96,
                              out_channels=256,
                              kernel_size=5,
                              padding=2).cuda()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 3rd Convolutional layer
        self.cnn3 = nn.Conv2d(in_channels=256,
                              out_channels=384,
                              kernel_size=3,
                              padding=1).cuda()

        # 4th Convolutional layer
        self.cnn4 = nn.Conv2d(in_channels=384,
                              out_channels=384,
                              kernel_size=3,
                              padding=1).cuda()

        # 5th Convolutional layer
        self.cnn5 = nn.Conv2d(in_channels=384,
                              out_channels=256,
                              kernel_size=3,
                              padding=1).cuda()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 6th Fully-connected layer
        self.fc1 = nn.Linear(in_features=9216, out_features=4096).cuda()

        # 7th Fully-connected layer
        self.fc2 = nn.Linear(in_features=4096, out_features=4096).cuda()

        # 8th Output layer
        self.fc3 = nn.Linear(in_features=4096, out_features=10).cuda()

    def forward(self, x):
        # 1st Convolution layer
        x = self.cnn1(x).cuda()
        x = torch.relu(x.cuda()).cuda()
        x = self.maxpool1(x.cuda()).cuda()

        # 2nd convolution layer
        x = self.cnn2(x).cuda()
        x = torch.relu(x.cuda()).cuda()
        x = self.maxpool2(x.cuda()).cuda()

        # 3rd convolution layer
        x = self.cnn3(x).cuda()
        x = torch.relu(x.cuda()).cuda()

        # 4th convolution layer
        x = self.cnn4(x.cuda()).cuda()
        x = torch.relu(x.cuda()).cuda()

        # 5th convolution layer
        x = self.cnn5(x.cuda()).cuda()
        x = torch.relu(x.cuda()).cuda()
        x = self.maxpool5(x).cuda()

        # Convert to a vector
        x = x.view(x.size(0), -1)

        # 6th fully-connected layer
        x = self.fc1(x).cuda()
        x = self.drop(x.cuda()).cuda()
        x = torch.relu(x.cuda()).cuda()

        # 7th output layer
        x = self.fc2(x).cuda()
        x = self.drop(x.cuda()).cuda()
        x = torch.relu(x.cuda()).cuda()

        # 8th output layer
        x = self.fc3(x.cuda()).cuda()

        return x


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def train(epoch_num, batch_size):
    target = (epoch_num * int(120000 / batch_size)) + 120000 - int(120000 / batch_size) * batch_size
    if torch.cuda.is_available():
        print("Using device: ", torch.cuda.get_device_name(0))
    else:
        print("Using device: CPU")

    train_result = {'training_loss': []}

    start_train = time.time()
    batch_count = 0
    for epoch in range(epoch_num):
        print('Epoch: {} / {}'.format((epoch + 1), epoch_num))

        for x, y in train_loader:
            start_time = time.time()
            x = torch.tensor(x).cuda()
            y = torch.tensor(y).cuda()
            optimizer.zero_grad()
            prediction = model(x)
            loss = cost_function(prediction, y).cuda()
            loss.backward()
            optimizer.step()
            train_result['training_loss'].append(loss.data.item())
            finish_time = time.time()
            total_time = finish_time - start_time
            remain = target - batch_count
            eta =  remain * total_time
            print('[{}/{}]  ---  {}%  ---  ETA: {}'.format(batch_count, int(target), round(((batch_count / target) * 100), 2), convert(eta)))
            batch_count += 1

        path = "/content/gdrive/My Drive/Alex Net"
        torch.save(model.state_dict(), path)
        print("Model saved!")
          
        
    finish_train = time.time()
    train_time = start_train - finish_train
            
    return train_result, train_time


def augment_data():
    affine = transforms.RandomAffine(degrees=(1, 180),
                                     translate=(0.2, 0.2),
                                     scale=(1, 1.5),
                                     shear=0.2)

    tf1 = transforms.Compose([transforms.Resize((227, 227)),
                              affine,
                              transforms.ToTensor()])

    tf2 = transforms.Compose([transforms.Resize((227, 227)),
                              transforms.ToTensor()])

    train_dataset1 = dsets.MNIST(root='./data', train=True, download=True, transform=tf1)
    train_dataset2 = dsets.MNIST(root='./data', train=True, download=True, transform=tf2)
    return train_dataset1 + train_dataset2


if __name__ == '__main__':
    # Resize the input image
    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    # NMIST dataset
    train_dataset = augment_data()
    validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=tf)

    # AlexNet model
    model = AlexNet(drop=0.5)
    model.cuda(0)
    model.train()

    # # Training parameters
    cost_function = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               pin_memory=True,
                                               num_workers=5)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                    batch_size=1000,
                                                    pin_memory=True,
                                                    num_workers=5)
    path = "/content/gdrive/My Drive/Alex Net"
    model.load_state_dict(torch.load(path))

    print("TRAINING STARTED")
    print("-----------------------------------")
    # train(epoch_num, batch_size)
    result, time = train(10, 128)
    model.eval()
    print("-----------------------------------")
    print("TRAINING COMPLETED")
    print("Time elapsed: " + convert(time))

    correct = 0
    for x, y in validation_loader:
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        z = model(x)
        _, prediction = torch.max(z.data, 1)
        correct += (prediction == y).sum().item()

    accuracy = 100 * (correct / len(validation_dataset))
    #print("Accuracy: " + str(accuracy) + "%" + "  ---  Time elapsed: " + convert(time))
    print("Accuracy: " + str(accuracy) + "%")

    # Show loss
    # plt.plot(result['training_loss'], label='relu')
    # plt.ylabel('Loss')
    # plt.title('Iterations')
    # plt.legend()
    # plt.show()
