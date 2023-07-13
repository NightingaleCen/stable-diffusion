import argparse, os, sys, datetime, glob, importlib, csv
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config

from ldm.data.artwork import ArtworkDataset


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        const=True,
        default=1 / 255,
        nargs="?",
        help="step size for PGD",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        const=True,
        default=100,
        nargs="?",
        help="number of PGD steps",
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        const=True,
        default=8 / 255,
        nargs="?",
        help="total perception budget of the perturbation",
    )
    # parser.add_argument(
    #    "-c",
    #    "--ckpt",
    #    type=str,
    #    const=True,
    #    required=True,
    #    nargs="?",
    #    help="path to the diffusion checkpoint",
    # )
    # temporarily abandoned
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        const=True,
        # required=True,
        default="images",  # TODO:remove
        nargs="?",
        help="path to the image dir",
    )
    parser.add_argument(
        "-w",
        "--watermark",
        type=str,
        const=True,
        # required=True,
        default="watermark.jpeg",  # TODO:remove
        nargs="?",
        help="path to the watermark file",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=512,
        help="resize the image to a certain size",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    return parser


class WatermarkPGD:
    def __init__(self, model, eps=8 / 255, alpha=1 / 255, iters=100):
        self.device = model.device
        self.model = model
        self.eps = eps * 2
        self.alpha = alpha * 2
        self.iters = iters
        self.cond = "a painting"

    def __call__(self, x):
        image = x["image"].clone().detach().to(self.device).requires_grad_(True)
        watermark = x["watermark"].clone().detach().to(self.device).requires_grad_(True)
        x = {"image": image, "watermark": watermark}
        loss_history = []

        for i in tqdm(range(self.iters)):
            loss = self.model(x, self.cond)
            loss_history.append(loss.item())
            self.model.zero_grad()
            loss.backward()

            image = x["image"] - self.alpha * x["image"].grad.sign()
            image = torch.min(
                torch.max(image, x["image"] - self.eps), x["image"] + self.eps
            )
            image = torch.clamp(image, -1, 1)
            x["image"] = image.detach().requires_grad_(True)

        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.savefig("curve.png")
        return x["image"]


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    seed_everything(opt.seed)

    # init config
    config = OmegaConf.load("configs/artwork-watermark/artwork-watermark.yaml")

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config.model).to(device)
    watermark = WatermarkPGD(model=model, eps=opt.eps, alpha=opt.alpha, iters=opt.iters)

    # data
    data = ArtworkDataset(
        data_root=opt.dir, watermark_path=opt.watermark, size=opt.size
    )
    dataloader = DataLoader(data, batch_size=opt.batchsize)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    # data.prepare_data()
    # data.setup()
    # print("#### Data #####")
    # for k in data.datasets:
    #    print(
    #        f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
    #    )

    # run
    for _, batch in enumerate(dataloader):
        result = watermark(batch)
        for i in range(result.shape[0]):
            img = result[i].detach().cpu().numpy()
            img = ((img + 1.0) * 127.5).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(f"{batch['file_name'][i]}_watermarked.{batch['extension'][i]}")
