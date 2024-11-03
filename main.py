import argparse

import torch
from torch.utils.data import random_split

from source.dataset import TrainScanImageDataset,ValidScanImageDataset, ScanImageTestDataset
from source.model import *
#from source.nafnet import *
from source.kbnet import *
from source.trainer import Trainer
from source.transform import get_test_transform, get_transform
from source.utils import set_seed, get_parameter_nums



def main(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transform = get_transform()
    test_transform = get_test_transform()

    model = Restormer()
    #model = Restormer(num_blocks=[1, 2, 2, 4], num_heads=[1, 1, 2, 4])
    #model = NAFNet(width=32, enc_blk_nums=[1, 1, 1, 12], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    #model = KBNet_l()
    

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> from resuming checkpoint '{args.resume}' ")

    train_dataset = TrainScanImageDataset(
        noisy_image_dir_path=args.train_noisy_image_dir, clean_image_dir_path=args.train_clean_image_dir, transform=transform
    )

    validation_dataset = TrainScanImageDataset(
        noisy_image_dir_path=args.valid_noisy_image_dir, clean_image_dir_path=args.valid_clean_image_dir, transform=transform
    )


    train_size = len(train_dataset)
    validation_size = len(validation_dataset)

    #train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    test_dataset = ScanImageTestDataset(noisy_image_paths=args.test_image_dir, transform=test_transform)

    trainer = Trainer(args, model, train_dataset, validation_dataset, test_dataset)
    print(f"Num parameters: {get_parameter_nums(model)}")

    if args.do_train:
        trainer.train()
    if args.do_eval:
        trainer.evaluate()
    if args.do_inference:
        trainer.inference(args.output_path, args.output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--split_ratio", default=0.95, type=float)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--num_train_epochs", default=1000, type=int)
    parser.add_argument("--train_noisy_image_dir", default=r"C:\Users\Admin\Downloads\event\Training\원천\훼손블러\imagedark\noisy", type=str)
    parser.add_argument("--train_clean_image_dir", default=r"C:\Users\Admin\Downloads\event\Training\원천\훼손블러\imagedark\01.GT", type=str)
    parser.add_argument("--valid_noisy_image_dir", default=r"C:\Users\Admin\Downloads\event\Validation\원천\훼손블러\어두운조도\noisy", type=str)
    parser.add_argument("--valid_clean_image_dir", default=r"C:\Users\Admin\Downloads\event\Validation\원천\훼손블러\어두운조도\01.원본_GT", type=str)
    parser.add_argument("--test_image_dir", default=r"C:\Users\Admin\Downloads\sub", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_wandb", default=True, type=bool)
    parser.add_argument("--do_eval", default=False, type=bool)
    parser.add_argument("--do_inference", default=True, type=bool)
    parser.add_argument("--dataset_path", default="dataset", type=str)
    parser.add_argument("--save_logs", default=True, type=bool)
    parser.add_argument("--save_frequency", default=5, type=int)
    parser.add_argument("--checkpoint_path", default="./model_output/", type=str)
    parser.add_argument("--submission_path", default="submission/", type=str)
    parser.add_argument("--output_path", default="output", type=str)
    parser.add_argument("--output_file_name", default="result.csv", type=str)
    parser.add_argument(
        "--resume",
        default=r'C:\Users\Admin\Desktop\UniDthon\best_11030620_epochs-1000_batch-8.pt',
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "--val_frequency", default=1, type=int, help="How often to run evaluation with validation data."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )

    args = parser.parse_args()

    main(args)
