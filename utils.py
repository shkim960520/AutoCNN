import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type = str,
    default = '../chest_xray',
    help = 'dataset path for training'
)

parser.add_argument(
    "--log_path",
    type = str,
    default = 'log',
    help = 'path for logging loss'
)

parser.add_argument(
    "--log_metrics",
    type = str2bool,
    default = "True",
    help = 'log metrics(True) or not(False)'
)

parser.add_argument(
    "--device_number",
    type = int,
    default = 0,
    help = "gpu number"
)
parser.add_argument(
    "--channel",
    type = int,
    default = 3,
    help = "number of channels of input image. It must be 1 or 3"
)
parser.add_argument(
    "--mode",
    type = str,
    default = 'auto',
    help = 'select model you want (auto or cnn)'
)

parser.add_argument(
    "--lr",
    type = float,
    default = 0.0001,
    help = "learning rate"
)

parser.add_argument(
    "--batch_size",
    type = int,
    default = 32,
    help = "batch size"
)

parser.add_argument(
    "--workers",
    type = int,
    default = 4,
    help = "number of workers for dataloader"
)

parser.add_argument(
    "--num_epoch",
    type = int,
    default = 30,
    help = "epoch"
)

parser.add_argument(
    "--resnet_layers",
    type = int,
    default = 50,
    help = 'number of resnet layers. It must be 50 or 101 or 152'
)

parser.add_argument(
    "--classes",
    type = int,
    default = 2,
    help = "number of classes you want to classify"
)
