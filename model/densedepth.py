import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width = 0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(skip_input=features//1 + 384, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 192, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 +  96, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  96, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self, pretrained=False, unlabeled=False):
        super(Encoder, self).__init__()
        self.original_model = torchvision.models.densenet161(pretrained=pretrained)
        self.unlabeled = unlabeled

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )

        if self.unlabeled:
            return features[-1]
        else:
            return features
    
class Model(nn.Module):
    def __init__(self, pretrained=False, layers=161):
        super(Model, self).__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.encoder.load_state_dict(torch.load('checkpoints/densenet_%d.pth'%layers))
        self.decoder = Decoder()
        

    def forward(self, x):
        return self.decoder( self.encoder(x) )


class Header(nn.Module):
    def __init__(self, num_classes=2):
        super(Header, self).__init__()

        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(2208, 256),
            nn.ReLU()
        )

    def forward(self, z):

        h = self.net(z)

        return h


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()

        self.module = module

    def forward(self, x):

        return self.module(x)



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=256, K=65536, m=0.999, T=0.07, layers=161):

        super(MoCo, self).__init__()

        self.K = int(K * (64/256))
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Encoder()
        self.encoder_k = Encoder()

        self.header_q = Header()
        self.header_k = Header()



        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.header_q.parameters(), self.header_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.header_q.parameters(), self.header_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)[-1]
        #print(len(q), q[0].shape)
        #q = q[-1]

        q = nn.AvgPool2d(15,20)(q).squeeze(dim=2)
        q = q.squeeze(dim=2)
        q = self.header_q(q) # queries: NxC
        #print(q.shape)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[-1]
            k = nn.AvgPool2d(15,20)(k).squeeze(dim=2)
            k = k.squeeze(dim=2)
            k = self.header_k(k) # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        clones = self.queue.clone().detach()
        #clones = clones.half()
        l_neg = torch.einsum('nc,ck->nk', [q, clones])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        labels= labels.cuda(None, non_blocking=True)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


if __name__ == '__main__':

    x = torch.ones(1,3,480,640)

    #models = torchvision.models.densenet169(pretrained=True)
    output = Encoder()(x)

    print(output[-1].shape)

