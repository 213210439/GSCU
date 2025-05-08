import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import MyDataset
from utils.visualize import Evaluate
from PanNet import PanNet, PanNet_Tconv,PanNet_nearest,PanNet_GSCU
from utils.metrics import ref_evaluate
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.visualize import Evaluate

def main():
    # 全局配置
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 300
    batch_size = 10
    exp_type = 'PanNet'
    data_type = 'WV2'
    evaluator = Evaluate(exp_type, data_type, device)
    # 数据集和数据加载器
    data_root = 'D:/桌面/毕业论文/数据/WV2_data/WV2_data'
    train_pan = 'train128/pan'
    train_ms = 'train128/ms'
    test_pan = 'test128/pan'
    test_ms = 'test128/ms'
    train_dataset = MyDataset(data_root, train_ms, train_pan, 'bicubic')
    test_dataset = MyDataset(data_root, test_ms, test_pan, 'bicubic')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型配置
    weight_decay = 1e-5
    learning_rate = 5e-4
    loss_fun = nn.MSELoss()
    model = PanNet_GSCU().to(device)
    optimizer = opt.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_loss = []
    test_loss = []

    # 训练循环
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0
        for label, pan, lrms, up_ms, hpan, hlrms in train_loader:
            label = torch.Tensor(label).to(device).float()
            pan = torch.Tensor(pan).to(device).float()
            lrms = torch.Tensor(lrms).to(device).float()
            hpan = torch.Tensor(hpan).to(device).float()
            # 前向传播
            out, up_ms = model(pan, lrms,  hpan)
            loss_1 = loss_fun(out, label)
            # optional: for residual structure
            loss_2 = loss_fun(up_ms, label)
            loss = loss_1 + loss_2

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss_1.item()

        # 计算平均loss
        train_loss.append(epoch_train_loss / train_loader.__len__())
        print('epoch:' + str(epoch),
              'Model train loss:' + str(epoch_train_loss / train_loader.__len__()))
        # 每10个epoch测试一次
        if epoch % 10 == 0:
            model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for label, pan, lrms, up_ms, hpan, hlrms in test_loader:
                    label = torch.Tensor(label).to(device).float()
                    pan = torch.Tensor(pan).to(device).float()
                    lrms = torch.Tensor(lrms).to(device).float()
                    hpan = torch.Tensor(hpan).to(device).float()

                    out,up_ms = model(pan, lrms, hpan)
                    loss = loss_fun(out, label)
                    epoch_test_loss += loss.item()

            test_loss.append(epoch_test_loss/test_loader.__len__())
            print('epoch:' + str(epoch),
                  'Model test loss:' + str(epoch_test_loss / test_loader.__len__()))

        # 可视化loss并保存模型
        evaluator.visualize(train_loss, test_loss, model)
        scheduler.step()

    print("\n=== 训练结束，输出最小损失值 ===")
    print(f"模型训练集最小损失: {min(train_loss) if train_loss else '无数据'}")
    print(f"模型测试集最小损失: {min(test_loss) if test_loss else '无数据'}")

    print("\n=== 加载最佳权重进行评估 ===")
    model = torch.load(evaluator.path + f'{exp_type}.pkl', map_location=device)
    model.eval()

    print("\n=== Evaluating Metrics ===")
    metrics = evaluator.evaluate_metrics(model, test_loader)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
