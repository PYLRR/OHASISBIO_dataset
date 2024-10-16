from torch import nn

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        modules.append(nn.Conv2d(1, 16, kernel_size=(7,7), padding='same'))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.MaxPool2d(kernel_size=(1,2)))

        modules.append(nn.Conv2d(16, 32, kernel_size=(7,7), padding='same'))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.MaxPool2d(kernel_size=(2,2)))

        modules.append(nn.Conv2d(32, 64, kernel_size=(7,7), padding='same'))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.MaxPool2d(kernel_size=(2,2)))

        modules.append(nn.Conv2d(64, 128, kernel_size=(3,3), padding='same'))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.MaxPool2d(kernel_size=(4,4)))

        modules.append(nn.Conv2d(128, 256, kernel_size=(4,4), padding='valid'))
        modules.append(nn.LeakyReLU(inplace=True))

        modules.append(nn.Flatten(-3, -1))

        modules.append(nn.Linear(256, 64))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.Linear(64, 32))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.Linear(32, 16))
        modules.append(nn.LeakyReLU(inplace=True))
        modules.append(nn.Linear(16, 16))
        modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class EmbedderSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        modules.append(nn.Conv2d(1, 16, kernel_size=(7, 7), padding='same'))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(16, 32, kernel_size=(7, 7), padding='same'))
        modules.append(nn.LeakyReLU())
        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(32, 64, kernel_size=(7, 7), padding='same'))
        modules.append(nn.LeakyReLU())
        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(64, 128, kernel_size=(3, 7), padding='same'))
        modules.append(nn.LeakyReLU())
        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(128, 256, kernel_size=(3, 7), padding='same'))
        modules.append(nn.LeakyReLU())
        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(256, 512, kernel_size=(4, 1), padding='valid'))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Flatten(-2, -1))

        modules.append(nn.Conv1d(512, 256, 7, padding='same'))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv1d(256, 64, 7, padding='same'))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv1d(64, 16, 7, padding='same'))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv1d(16, 16, 7, padding='same'))

        modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)