import argparse
from model import Flow
import torch
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Flow Based Generative Model")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epoch", type=int, default=25000)
    parser.add_argument("-b", "--batch", type=int, default=20)
    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow = Flow().to(device)
    flow.load(lr=args.learning_rate)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(
        './data',
        train = True,
        download = True,
        transform = transform
        )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch,
        shuffle = True)
    
    writer = SummaryWriter()
    for i in range(args.epoch):
        for j, (x, _) in enumerate(tqdm(dataloader)):
            flow.train()
            x = x.to(device)
            noise = torch.distributions.Uniform(0., 1.).sample(x.size()).to(device)
            x = (x*255. + noise) /256.
            flow.one_epoch(x)
            if j % 100 == 0:
                flow.eval()
                img = flow.gen(64)
                grid = utils.make_grid(img)
                writer.add_image("images", grid, j)
        