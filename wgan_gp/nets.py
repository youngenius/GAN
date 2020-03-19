import torch
import torch.nn as nn
import torch.nn.functional as F
'''
class ResUnit(nn.Module):
    def __init__(self):
        super(ResUnit, self).__init__()



    def forward(self, x):
        x2 = F.relu(self.conv1(x))
        return F.relu(self.conv2(x2)) + x
'''
class Generator(nn.Module):
    def __init__(self,flags):
        super(Generator, self).__init__()
        self.flags = flags
        self.image_shape = (flags.channels, flags.img_size, flags.img_size)

        self.l1 = nn.Linear(flags.latent_dim,128)
        self.l2 = nn.Linear(128, 128 * 4 * 4)
        #self.bn1 = nn.BatchNorm1d(128)
        #self.bn1_2 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv1= nn.Conv2d(128, 128, 3, padding=1)
        self.conv2= nn.Conv2d(128, 3, 3, padding=1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        '''
        def ResUnit(in_feat, out_feat, normalize=True):
            layers = nn.Conv2d(in_feat, out_feat, kernel_size=3)
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat), 0.8)
            layers.append(nn.ReLU(0.2,inplace=True))
            return layers
        '''
        '''
        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat), 0.8)
            layers.append(nn.ReLU(0.2, inplace=True))
            return layers

        def last(in_feat, out_feat, normalize =True):
            layers = nn.Conv2d(in_feat,out_feat, kernel_size=3)
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat),0.8)
            layers.append(nn.Tanh())
            return layers


        self.model = nn.Sequential(
            *block(flags.latent_dim, 128, normalize=False),
            *block(128, 128*4*4),
            ResUnit(128,128, 'upsampling'), #4*4->8*8
            ResUnit(128,128, 'upsampling'), #8*8->16*16
            ResUnit(128,128, 'upsampling'), #16*16->32*32
            ResUnit(128,128, 'upsampling'), #32*32->64*64
            ResUnit(128,128, 'upsampling'), #64*64->128*128
            ResUnit(128,128, 'upsampling'), #128*128->256*256
            last(128,3)
        )
        '''
    def forward(self, x):
        #print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        #print(x.shape)
        x = x.view(self.flags.batch,128,4,4)

        #residual block
        x2 = F.relu(self.bn2(self.conv1(x))) # 4*4
        x3 = F.relu(self.bn2(self.conv1(x2))) # 4*4
        x4 = x3+x
        x5 = self.upsampling(x4) # 8*8

        #residual block
        x6 = F.relu(self.bn2(self.conv1(x5))) # 8*8
        x7 = F.relu(self.bn2(self.conv1(x6))) #8*8
        x8 = x5+x7
        x9 = self.upsampling(x8) #16*16

        # residual block
        x10 = F.relu(self.bn2(self.conv1(x9)))  # 16*16
        x11 = F.relu(self.bn2(self.conv1(x10)))  # 16*16
        x12 = x9 + x11
        x13 = self.upsampling(x12)  # 32*32

        # residual block
        x14 = F.relu(self.bn2(self.conv1(x13)))  # 32*32
        x15 = F.relu(self.bn2(self.conv1(x14)))  # 32*32
        x16 = x13 + x15
        x17 = self.upsampling(x16)  # 64*64

        # residual block
        x18 = F.relu(self.bn2(self.conv1(x17)))  # 64*64
        x19 = F.relu(self.bn2(self.conv1(x18)))  # 64*64
        x20 = x17 + x19
        x21 = self.upsampling(x20)  # 128*128

        # residual block
        x22 = F.relu(self.bn2(self.conv1(x21)))  # 128*128
        x23 = F.relu(self.bn2(self.conv1(x22)))  # 128*128
        x24 = x21 + x23
        x25 = self.upsampling(x24)  # 256*256

        x26 = F.relu(self.conv2(x25))
        x27 = F.tanh(x26)
        #img = x27.view(x27.size(0), self.image_shape)
        #print(x27.shape)
        '''
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape)
        '''
        return x27

class Discriminator(nn.Module):
    def __init__(self, flags):
        super(Discriminator, self).__init__()
        self.flags = flags
        self.l1 = nn.Linear(128, 1)
        self.conv1 = nn.Conv2d(128,128,3, padding=1)
        self.conv_input = nn.Conv2d(3,128,3,padding=1)
        self.meanpooling = nn.AvgPool2d(3,2,1)
        self.meanpooling2= nn.AvgPool2d(8,2)

    def forward(self, x):
        #input shape 맞춰주기
        input = F.leaky_relu(self.conv_input(x))
        #print(input.shape)
        #resunit
        x2 = F.leaky_relu(self.conv1(input))
        x3 = F.leaky_relu(self.conv1(x2))
        x4 = input+x3
        x5 = self.meanpooling(x4) # 128*128
        #print(x5.shape)
        #resunit
        x6 = F.leaky_relu(self.conv1(x5))
        x7 = F.leaky_relu(self.conv1(x6))
        x8 = x7 + x5
        x9 = self.meanpooling(x8)  # 64*64

        # resunit
        x10 = F.leaky_relu(self.conv1(x9))
        x11 = F.leaky_relu(self.conv1(x10))
        x12 = x9 + x11
        x13 = self.meanpooling(x12)  # 32*32

        # resunit
        x14 = F.leaky_relu(self.conv1(x13))
        x15 = F.leaky_relu(self.conv1(x14))
        x16 = x15 + x13
        x17 = self.meanpooling(x16)  # 16*16

        # resunit
        x18 = F.leaky_relu(self.conv1(x17))
        x19 = F.leaky_relu(self.conv1(x18))
        x20 = x19 + x17
        x21 = self.meanpooling(x20)  # 8*8

        # resunit
        x22 = F.leaky_relu(self.conv1(x21))
        x23 = F.leaky_relu(self.conv1(x22))#8*8
        x24 = x23 + x21

        # resunit
        x25 = F.leaky_relu(self.conv1(x24))
        x26 = F.leaky_relu(self.conv1(x25))#8*8
        x27 = x26 + x24 # [32,128,1,1]

        #x28 = F.relu(self.meanpooling2(x27)) # 1

        x28 = x27.view(self.flags.batch, 128)
        #print(x28.shape)
        x29 = self.l1(x28)

        return x29