import os
import json

import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import VisionTransformer

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = args.img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
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
    # load model weights
    model_weight_path = args.model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.savefig("test_result.png")
    #plt.show()


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

    parser.add_argument('--img_path', type=str, default="../test.jpg", help="测试图片路径")
    parser.add_argument('--model_path', type=str, default="weights/model-9.pth")
    args = parser.parse_args()
    test()