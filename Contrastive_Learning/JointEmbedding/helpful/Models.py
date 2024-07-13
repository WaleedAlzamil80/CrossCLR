import torch.nn as nn


class JointModel(nn.Module):
    def __init__(self):
        super(JointModel, self).__init__()
        self.LatentSpace = nn.Linear(1536, 128)

    def forward(self, X):
        return self.LatentSpace(X)

class EncoderMinist(nn.Module):
  def __init__(self):
    super().__init__()

    # Build and define the Encoder
    self.encoder = nn.Sequential(

        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = "same"),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding = "same"),
        nn.ReLU()
        )
    self.InLatentSpace = nn.Sequential(
        nn.Linear(in_features = 28 * 28 * 128, out_features = 512)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.reshape((-1, 128 * 28 * 28))
    x = self.InLatentSpace(x)
    return(x)