import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1 = nn.Sequential()
        self.hidden_layer2 = nn.Sequential()
        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = nn.ConvTranspose2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer
                nn.init.normal(input_deconv.weight, mean=0.0, std=0.02)
                nn.init.constant(input_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer1.add_module('input_bn', nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer1.add_module('input_act', nn.ReLU())

                # For label
                label_deconv = nn.ConvTranspose2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                nn.init.normal(label_deconv.weight, mean=0.0, std=0.02)
                nn.init.constant(label_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer2.add_module('label_act', nn.ReLU())
            else:
                deconv = nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                nn.init.normal(deconv.weight, mean=0.0, std=0.02)
                nn.init.constant(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, nn.ReLU())

        # Output layer
        self.output_layer = nn.Sequential()
        # Deconvolutional layer
        out = nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        nn.init.normal(out.weight, mean=0.0, std=0.02)
        nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return 
		
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer1 = nn.Sequential()
        self.hidden_layer2 = nn.Sequential()
        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = nn.Conv2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                nn.init.normal(input_conv.weight, mean=0.0, std=0.02)
                nn.init.constant(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('input_act', nn.LeakyReLU(0.2))

                # For label
                label_conv = nn.Conv2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Initializer
                nn.init.normal(label_conv.weight, mean=0.0, std=0.02)
                nn.init.constant(label_conv.bias, 0.0)

                # Activation
                self.hidden_layer2.add_module('label_act', nn.LeakyReLU(0.2))
            else:
                conv = nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                nn.init.normal(conv.weight, mean=0.0, std=0.02)
                nn.init.constant(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = nn.Sequential()
        # Convolutional layer
        out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        nn.init.normal(out.weight, mean=0.0, std=0.02)
        nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', nn.Sigmoid())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out
