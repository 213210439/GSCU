import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm
from utils.metrics import ref_evaluate


class Evaluate:
    def __init__(self, exp_type, data_type, device):
        # 初始化实验路径
        self.path = f'result/{exp_type}/{data_type}'
        self.data_type = data_type
        self.exp_type = exp_type
        self.device = device
        self.best_loss = 1
        i = 0
        while os.path.exists(self.path + str(i)):
            i += 1
        self.path = self.path + str(i) + '/'
        os.makedirs(self.path)

    def visualize(self, train_loss, test_loss, model):
        # 保存最低loss的模型权重
        if train_loss[-1] < self.best_loss:
            self.best_loss = train_loss[-1]
            torch.save(model, self.path + f'{self.exp_type}.pkl')

        # 绘制loss曲线
        plt.figure()
        plt.grid(color='#7d7f7c', linestyle='-.')
        plt.plot(np.arange(len(train_loss)), train_loss, 'c', linewidth=1.5, label=f"train {self.exp_type}")
        plt.plot(np.arange(0, len(train_loss), 10), test_loss, 'c--', linewidth=1.5, label=f"test {self.exp_type}")
        plt.title(f'Loss: {self.exp_type}')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        ylim = {'GF2': 1e-4, 'WV2': 5e-4, 'WV3': 1e-2}
        plt.ylim(0, ylim[self.data_type[:3]])
        plt.savefig(self.path + 'loss.jpg', dpi=300)
        plt.close()

    def evaluate_metrics(self, model, test_loader):
        # 在测试集上计算评价指标
        model.eval()
        metrics = {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'SCC': [], 'Q': []}
        metric_names = ['PSNR', 'SSIM', 'SAM', 'ERGAS', 'SCC', 'Q']

        with torch.no_grad():
            for label, pan, lrms, up_ms, hpan, hlrms in tqdm(test_loader, desc="Evaluating"):
                label = torch.Tensor(label).to(self.device).float()
                pan = torch.Tensor(pan).to(self.device).float()
                lrms = torch.Tensor(lrms).to(self.device).float()
                hpan = torch.Tensor(hpan).to(self.device).float()
                # 模型推理
                pred,_ = model(pan, lrms, hpan)
                pred_np = pred.cpu().numpy()
                label_np = label.cpu().numpy()
                # 计算每张图像的指标
                for i in range(pred_np.shape[0]):
                    pred_img = pred_np[i].transpose(1, 2, 0)
                    gt_img = label_np[i].transpose(1, 2, 0)
                    if pred_img.max() <= 1.0:
                        pred_img = (pred_img * 255).astype(np.uint8)
                        gt_img = (gt_img * 255).astype(np.uint8)
                    metric_values = ref_evaluate(pred_img, gt_img)
                    for name, value in zip(metric_names, metric_values):
                        metrics[name].append(value)

        # 计算平均指标并保存
        results = {}
        with open(self.path + 'evaluation_results.txt', 'w') as f:
            f.write(f"=== Evaluation Results for {self.exp_type} ===\n")
            for name in metric_names:
                avg_metric = np.mean(metrics[name])
                results[name] = avg_metric
                f.write(f"{name}: {avg_metric:.4f}\n")
        return results
