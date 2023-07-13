class resmnist(nn.Module):
    def __init__(self,in_chan=1):
        super(resmnist,self).__init__()
        self.model = md.resnet18(pretrained = True) # prebuillt resent 18 model
        self.model.conv1 = nn.Conv2d(in_chan, 64,kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model.drop = nn.Dropout2d(0.4) #droput function
        
        
        
    def forward(self, x):
        return self.model(x)
