import argparse
from dataset import *
import torch
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from nets2 import Generator, Discriminator
from torch.autograd import Variable
import numpy as np
from torch import autograd

def get_gradient(prob_interpolated, interpolated): #gradient자체가 loss
    gradient = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                             grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                             create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * flags.lamda
    return grad_penalty

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

def get_interpolated_samples(real, gen, flags):
    eta = torch.FloatTensor(flags.batch, 1, 1, 1).uniform_(0,1)
    #eta = torch.rand(flags.batch, 1, 1, 1)
    eta = eta.expand(flags.batch, real.size(1), real.size(2), real.size(3))
    eta = eta.cuda()

    interpolated = eta * real + ((1-eta)*gen)
    return interpolated

def main(flags):
    # 각도와 상관없이 다른각도의 얼굴을 생성하고 출력할때 조건을 앞으로 주면됨
    # loss를 같은사람이게끔 인식하게 해야함.
    # a 사람인지 b사림인지 판단 -> classification ?
    cuda = True if torch.cuda.is_available() else False
    ds = Dataset(flags)

    #Initialize generator and discriminator
    G = Generator(flags)
    D = Discriminator(flags)
    if cuda:
        G.cuda()
        D.cuda()

    learning_rate = 1e-4
    b1 = 0.5 #0
    b2 = 0.999

    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=(b1,b2))
    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=(b1,b2))

    dir_name = flags.model + time.strftime('~%Y%m%d~%H%M%S', time.localtime(time.time()))
    log_train = './log/' + dir_name + '/train'
    writer = SummaryWriter(log_train)

    generator_iters = flags.epoch
    critic_iters = flags.n_critic

    #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    data = get_infinite_batches(ds.load_dataset())

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()

    for g_iter in range(generator_iters):
        # Requires grad, Generator requires_grad = False

        for p in D.parameters():
            p.requires_grad = True

        for d_iter in range(critic_iters):
            D.zero_grad()
            imgs = data.__next__()
            #input
            #real_imgs = Variable(imgs.type(Tensor))
            # sample noise as generator input
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], flags.latent_dim))))
            z = torch.rand((flags.batch, 100, 1, 1))
            real_imgs, z = Variable(imgs.cuda()), Variable(z.cuda())

            #train discriminator
            #d_optimizer.zero_grad()
            d_real_loss = D(real_imgs)
            d_real_loss = d_real_loss.mean()
            d_real_loss.backward(mone)
            # Generate a batch of images
            gen_imgs = G(z)
            d_fake_loss = D(gen_imgs)
            d_fake_loss = d_fake_loss.mean()
            d_fake_loss.backward(one)

            interpolated_samples = get_interpolated_samples(real_imgs, gen_imgs, flags)
            interpolated_samples = interpolated_samples.cuda()
            interpolated_samples = Variable(interpolated_samples, requires_grad= True)
            prob_interpolated = D(interpolated_samples)
            grad_penalty = get_gradient(prob_interpolated, interpolated_samples)
            grad_penalty.backward(one)
            #print(d_fake_loss, d_real_loss, grad_penalty)
            discriminator_loss = d_fake_loss - d_real_loss + grad_penalty
            #discriminator_loss.backward()
            # measure discriminator's ability to classify real from generated samples
            wasserstein_distance = d_real_loss - d_fake_loss
            #discriminator_loss.backward() ?? 이거는 왜 또 안해
            d_optimizer.step()
        #train generator
        for p in D.parameters():
            p.requires_grad = False

        G.zero_grad()
        #g_optimizer.zero_grad()

        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], flags.latent_dim))))
        z = Variable(torch.randn(flags.batch, 100, 1, 1)).cuda()
        fake_images = G(z)
        d_fake_loss = D(fake_images)
        generator_loss = d_fake_loss.mean()#mean까지 한 결과여야함
        generator_loss.backward(mone)
        g_optimizer.step()
        generator_loss = -generator_loss
        print("[step: %10d] d_loss: %f, g_loss: %f, wasserstein_d: %f" % (g_iter, discriminator_loss, generator_loss, wasserstein_distance))
        if (g_iter)% 100 == 0:
            #print("[step: %10d] d_loss: %f, g_loss: %f, wasserstein_d: %f" % (g_iter, discriminator_loss, generator_loss, wasserstein_distance))
            #tensorboard logging
            writer.add_scalar('d_loss', discriminator_loss, g_iter*5)
            writer.add_scalar('g_loss', generator_loss, g_iter*5)
            writer.add_scalar('wasserstein_distance', wasserstein_distance, g_iter*5)
            writer.add_images('fake_image', fake_images, g_iter*5)
            #writer.add_images('real_image', real_imgs, g_iter*5)
    '''
    for step, (batches,target) in enumerate(ds.load_dataset()):
        for batch in batches:
            plt.imshow(batch.permute(1,2,0))
            plt.show()
    '''
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gan_model",
        epilog="dd"
    )
    parser.add_argument('--model', type=str, default="wgan-gp",help="model save name")
    parser.add_argument('--data', type=str, default="/home/yjheo/dataset", help="face dataset")
    parser.add_argument('--batch', type=int, default=32, help="batch size")
    parser.add_argument('--epoch', type=int, default=30000, help="epoch")
    parser.add_argument('--latent_dim', type=int, default=100, help="latent dimension")
    parser.add_argument('--channels', type=int, default=3, help="image channel")
    parser.add_argument('--img_size', type=int, default=64, help="image size")
    parser.add_argument('--n_critic', type=int, default=5, help="default critic num")
    parser.add_argument('--lamda', type=int, default=10, help='default lamda')
    flags, _=parser.parse_known_args()
    main(flags)