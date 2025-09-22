from .clip import clip 
from PIL import Image
import torch
import torch.nn as nn
import random


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, opt, num_classes=1):
        super(CLIPModel, self).__init__()

        self.layer = str(22)
        print("Choose layer:", self.layer, "for cls embedding")

        self.randomErasing = False
        self.b_low = 0.03  
        self.b_high = 0.3  
        self.erase_prob = 0.1

        self.addNoise = False
        self.NoiseSTD = 0.01
        if opt.isTrain and opt.GaussianNoise:
            self.addNoise = True
            print("Add Gaussian noise to the feature embedding when training with std:", self.NoiseSTD)
        else:
            print("Not add Gaussian noise to the feature embedding.")
        if opt.isTrain and opt.RandomErasing:
            self.randomErasing = True
            print("Random erase the feature embedding with ratio:[{0},{1}] and prob:{2}".format(self.b_low, self.b_high, self.erase_prob))
        else:
            print("Not use random erasing.")

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        if(self.layer != "final"):
            # self.fc = nn.Linear( 1024, num_classes)
            self.fc = nn.Sequential(nn.Linear(1024, 10),  
                nn.LeakyReLU(0.01),  
                nn.Dropout(p=0.3),
                nn.Linear(10, num_classes)  
                )
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
        else:
            self.fc = nn.Linear( CHANNELS[name], num_classes)
    
    def random_vector_erase(self, features, erase_ratio=0.1):
        batch_size, feature_dim = features.shape
        erase_count = int(feature_dim * erase_ratio)

        for i in range(batch_size):
            erase_indices = torch.randperm(feature_dim)[:erase_count]
            features[i][erase_indices] = 0

        return features

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x, evaling=False, return_feature=False):
        features = self.model.encode_image(x, self.layer) 

        if self.addNoise and not evaling:
            noise = torch.randn_like(features) * self.NoiseSTD  
            features = features + noise

        if self.randomErasing and not evaling and random.random() > self.erase_prob:
            erase_ratio = random.uniform(self.b_low, self.b_high)  
            features = self.random_vector_erase(features, erase_ratio=erase_ratio)

        if return_feature:
            return features
            
        return self.fc(features)

