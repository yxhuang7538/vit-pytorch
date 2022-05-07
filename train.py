import os
import math
import argparse
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets import MyDataSet
from vit_model import VisionTransformer
from utils import read_split_data, train_one_epoch, evaluate

def train(args):
    """训练模型"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = VisionTransformer(args.img_size, 
                              args.patch_size,
                              args.in_c,
                              args.num_classes,
                              args.embed_dim,
                              args.depth,
                              args.num_heads,
                              args.mlp_ratio,
                              args.qkv_bias,
                              args.qk_scale,
                              args.representation_size,
                              args.drop_ratio,
                              args.attn_drop_ratio).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 网络参数
    parser.add_argument('--img_size', type=int, default=224, help="输入图像大小")
    parser.add_argument('--patch_size', type=int, default=16, help="分割的patch大小")
    parser.add_argument('--in_c', type=int, default=3, help="输入图像通道数")
    parser.add_argument('--num_classes', type=int, default=5, help="类别数量")
    parser.add_argument('--embed_dim', type=int, default=768, help="编码维度，16x16x3=768")
    parser.add_argument('--depth', type=int, default=12, help="tf的encoder堆叠层数")
    parser.add_argument('--num_heads', type=int, default=12, help="tf的注意力头数量")
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help="MLP结构中的膨胀系数")
    parser.add_argument('--qkv_bias', type=bool, default=True, help="启用qkv偏置功能")
    parser.add_argument('--qk_scale', type=int, default=None, help="计算qk分数时的分母缩放系数")
    parser.add_argument('--representation_size', type=int, default=None, help="重新表征大小")
    parser.add_argument('--drop_ratio', type=float, default=0., help="除attention层外其余层的丢弃概率")
    parser.add_argument('--attn_drop_ratio', type=float, default=0., help="attention层中的丢弃概率")
    parser.add_argument('--epochs', type=int, default=10, help="训练批次")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="../fruits", help="数据集位置")
    parser.add_argument('--model-name', default='', help="模型名字")
    parser.add_argument('--weights', type=str, default='weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='初始化权重路径')
    parser.add_argument('--freeze-layers', type=bool, default=True, help="是否冻结权重")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    train(args)