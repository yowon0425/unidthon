import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from .utils import *
from .loss import * 
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class Trainer(object):
    def __init__(self, args, model=None, train_dataset=None, valid_dataset=None, test_dataset=None):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        

    def train(self):
        if self.args.do_wandb:
            kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
            name = kor_time + "_epochs-" + str(self.args.num_train_epochs) + "_batch-" + str(self.args.batch_size)
            wandb.init(
                project="UniDthon-4th",
                entity="UniDthon-4th",
                name=name,
                config={
                    "learning_rate": self.args.learning_rate,
                    "epochs": self.args.num_train_epochs,
                    "batch_size": self.args.batch_size,
                },
            )
        #combined_dataset = ConcatDataset([self.train_dataset, self.valid_dataset])
        #train_dataloader = DataLoader(combined_dataset,self.args.batch_size,num_workers=self.args.num_workers, shuffle=True)
        
        train_dataloader = DataLoader(self.train_dataset, self.args.batch_size, num_workers=self.args.num_workers)
        valid_dataloader = DataLoader(self.valid_dataset, self.args.eval_batch_size, num_workers=self.args.num_workers)

        total_steps = len(train_dataloader) * self.args.num_train_epochs
        optimizer = optim.AdamW(self.model.parameters(),weight_decay=1e-4)
        
        #scheduler = CosineAnnealingLR(optimizer, T_max=20)
        
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        
        self.model.train()
        for epoch in range(self.args.num_train_epochs):
            step = 0
            train_loss = 0.0
            best_loss = 9999.0
            pbar = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
            for noisy_images, clean_images in pbar:
                optimizer.zero_grad()
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)
                with torch.autocast(device_type=self.device):
                    outputs = self.model(noisy_images)
                    loss = self.criterion(clean_images,outputs)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                train_loss += loss.item() * noisy_images.size(0)
                #scheduler.step()
                pbar.set_description(f"epoch: {epoch}/ train loss: {loss.item()}")
                if self.args.do_wandb:
                    wandb.log({"train_loss": loss.item(), "train_epoch": epoch, "lr": get_lr(optimizer)})

            if self.args.save_logs:
                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            train_loss /= step
            if self.args.val_frequency > 0 and (epoch + 1) % self.args.val_frequency == 0:
                self.evaluate()
                
            if best_loss > train_loss:
                best_loss = train_loss
                model_name = f"best_{name}.pt"
                torch.save(
                    checkpoint_dict,
                    os.path.join(self.args.checkpoint_path, model_name),
                )

            if epoch + 1 == self.args.num_train_epochs or (
                self.args.save_frequency > 0 and ((epoch + 1) % self.args.save_frequency) == 0
            ):
                model_name = f"{name}_{epoch}.pt"

                torch.save(
                    checkpoint_dict,
                    os.path.join(self.args.checkpoint_path, model_name),
                )

    def evaluate(self):
        valid_dataloader = DataLoader(self.valid_dataset, self.args.eval_batch_size, num_workers=self.args.num_workers)
        self.model.eval()
        pbar = tqdm(valid_dataloader, total=len(valid_dataloader), leave=True)
        eval_loss = 0
        num_samples = 0
        content_loss = ContentLoss(torch.nn.MSELoss())
        content_loss.eval()
        with torch.no_grad():
            for noisy_images, clean_images in pbar:
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)
                with torch.autocast(device_type=self.device):
                    outputs = self.model(noisy_images)
                    loss = self.criterion(clean_images,outputs)

                batch_size = noisy_images.size(0)
                eval_loss += loss * batch_size
                num_samples += batch_size
                pbar.set_description(f"validation loss: {loss.item()}")
            eval_loss = eval_loss / num_samples
            print(f"validation loss: {eval_loss}")
            if self.args.do_wandb:
                wandb.log({"valid_loss": eval_loss.item()})
    
    def inference(self, output_path, output_file_name):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        pbar = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
        self.model.eval()
        for noisy_image, noisy_image_path in pbar:
            noisy_image = noisy_image.to(self.device)
            with torch.cuda.amp.autocast():
                denoised_image = self.model(noisy_image)
                #denoised_image = noisy_image - noise
            denoised_image = denoised_image.cpu().squeeze(0)
            denoised_image = torch.clamp(denoised_image, 0, 1)  # 이미지 값을 0과 1 사이로 클램핑
            denoised_image = torchvision.transforms.ToPILImage()(denoised_image)
            output_filename = noisy_image_path[0]
            denoised_filename = output_path + "/" + output_filename.split("/")[-1][:-4] + ".png"
            denoised_image.save(denoised_filename)
        '''
        file_names = os.listdir(output_path)
        file_names.sort()
        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time + "_epochs-" + str(self.args.num_train_epochs) + "_batch-" + str(self.args.batch_size)
        output_file_name = name + "-" + output_file_name
        submission_file = os.path.join(self.args.submission_path, output_file_name)
        with open(submission_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image File", "Y Channel Value"])

            for file_name in file_names:
                # 이미지 로드
                image_path = os.path.join(output_path, file_name)
                image = cv2.imread(image_path)

                # 이미지를 YUV 색 공간으로 변환
                image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                # Y 채널 추출
                y_channel = image_yuv[:, :, 0]

                # Y 채널을 1차원 배열로 변환
                y_values = np.mean(y_channel.flatten())

                # 파일 이름과 Y 채널 값을 CSV 파일에 작성
                writer.writerow([file_name[:-4], y_values])

        print("CSV file created successfully.")
        '''
