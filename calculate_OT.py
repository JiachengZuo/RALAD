import numpy as np
import torch
import torch.nn as nn
import sys
from torchvision import transforms
import os
from PIL import Image
import PIL.Image as pil
from torchvision.transforms import ToPILImage
from crossView import model
from crossView import model, CrossViewTransformer, CycledViewProjection
import argparse
from easydict import EasyDict as edict
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import tqdm
from losses import compute_losses, compute_losses_fine_turn
import crossView
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from utils import mean_IU, mean_precision
import random
import glob
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def get_args():
    parser = argparse.ArgumentParser(
        description="Fine_Turn options")
    parser.add_argument("--model_path", type=str, 
                        default='./ckpts',
                        help="path to MonoLayout model")
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        # default="",
        default="./ckpts",
        # default="./models/3Dobject/crossView/weights_200",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--lr", type=float, default=1e-4,  # attention
                        help="learning rate")
    parser.add_argument("--lr_transform", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--lr_steps', default=[200, 400], type=float, nargs="+",  # attention
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Max number of training epochs")
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="extension of images in the folder")
    parser.add_argument("--out_dir", type=str,
                        default="./outputs")
    parser.add_argument("--view", type=str, default=1, help="view number")
    parser.add_argument(
        "--split",
        type=str,
        default="3Dobject",
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw"],
        help="Data split for training/validation")
    parser.add_argument("--data_path", type=str, default="./datasets/kitti/object/training",
                        choices=[
                            './datasets/argoverse',
                            './datasets/kitti/object/training',
                            './datasets/kitti/odometry',
                            './datasets/kitti/raw'],
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")
    
    parser.add_argument("--model_name", type=str, default="crossView",
                        help="Model Name with specifications")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        default="dynamic",
        choices=[
            "both",
            "static",
            "dynamic"],
        help="Type of model being trained")
    parser.add_argument("--global_seed", type=int, default=0,
                        help="seed")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Mini-Batch size")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                        help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                        help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument('--log_root', type=str, default=os.getcwd() + '/log')
    parser.add_argument('--model_split_save', type=bool, default=True)

    configs = edict(vars(parser.parse_args()))
    return configs

def sinkhorn_distance(r, c, M, reg=1e-2, error_thres=1e-5, niter=100):
    device = r.device

    b, m, n = M.shape
    assert r.shape[0] == c.shape[0] == b and r.shape[1] == m and c.shape[1] == n, "r.shape=%s, c=shape=%s, M.shape=%s" % (r.shape, c.shape, M.shape)

    K = (-M / reg).exp()  # (b, m, n)
    u = torch.ones_like(r) / m  # (b, m)
    v = torch.ones_like(c) / n  # (b, n)

    for _ in range(niter):
        r0 = u
        # 避免出现除以0的情况
        u = r / (torch.einsum("bmn,bn->bm", [K, v]) + 1e-10)  # (b, m)
        v = c / (torch.einsum("bmn,bm->bn", [K, u]) + 1e-10)  # (b, n)

        err = (u - r0).abs().mean()
        if err.item() < error_thres:
            break

    T = torch.einsum("bm,bn->bmn", [u, v]) * K
    return T

def calculate_dot_feature(feature_1, feature_2, beta):
    b, c, _ , _= feature_1.shape
    feature_1 = feature_1.float()
    feature_2 = feature_2.float()
    feature_1 = feature_1.flatten()
    feature_2 = feature_2.flatten()
    feature_1 = feature_1.reshape(b, c, -1)
    feature_2 = feature_2.reshape(b, c, -1)

    l = feature_1.shape[-1]
    dist = (torch.arange(l).view(l, 1) - torch.arange(l).view(1, l)).abs().float()
    D = dist / dist.max()
    D = D.cpu().numpy()
    _M = torch.from_numpy(D).float().unsqueeze(0)  # b, m, n
    # 保存最好的id
    result_save = []

    for i in range(feature_1.shape[0]): # Feature_1 batch循环
        result_dot = sys.maxsize
        index = 0
        for j in range(feature_2.shape[0]): # Feature_2 batch循环
            result_tmp = 0
            sinkhorn_sum = 0
            l2_sum = 0
            # 计算整张图的sinkhorn distance和l2 distance
            for k in range(feature_1.shape[1]): # 通道循环
                # 计算sinkhorn_distance
                P = sinkhorn_distance(feature_1[i][k].unsqueeze(0), feature_2[j][k].unsqueeze(0), _M,reg=1e-2)
                P = P.squeeze(0).detach().cpu().numpy()
                T = P * D
                sinkhorn_sum = np.sum(T)
                # 计算l2距离的平方
                squared_diff = (feature_1[i][k] - feature_2[j][k]) ** 2
                l2_sum = torch.sum(squared_diff)

                # 计算dot
                result_tmp += torch.sum(l2_sum * sinkhorn_sum) + beta * sinkhorn_sum * torch.log(torch.tensor(sinkhorn_sum))
            # 保存最优的结果
            # 计算dot
            if result_tmp < result_dot:
                # 更新最好的结果，并保存index
                result_dot = result_tmp
                index = j

        #合并结果
        result_save.append(feature_1[i] + feature_2[index])

    return result_save

def calculate_dot(feature_1, feature_2, beta):
    l = feature_1.shape[-1]
    dist = (torch.arange(l).view(l, 1) - torch.arange(l).view(1, l)).abs().float()
    D = dist / dist.max()
    D = D.cpu().numpy()
    _M = torch.from_numpy(D).float().unsqueeze(0)  # b, m, n

    # 计算sinkhorn_distance
    P = sinkhorn_distance(feature_1, feature_2, _M,reg=1e-2)
    P = P.squeeze(0).detach().cpu().numpy()
    T = P * D
    sinkhorn_sum = np.sum(T)
    # 计算l2距离的平方
    squared_diff = (feature_1 - feature_2) ** 2
    l2_sum = torch.sum(squared_diff)

    # 计算dot
    return torch.sum(l2_sum * sinkhorn_sum) + beta * sinkhorn_sum * torch.log(torch.tensor(sinkhorn_sum))

def calculate_dot_feature_cuda(feature_1, feature_2, beta):
    b, c, _, _ = feature_1.shape
    feature_1 = feature_1.float().cuda()
    feature_2 = feature_2.float().cuda()
    feature_1 = feature_1.flatten()
    feature_2 = feature_2.flatten()
    feature_1 = feature_1.reshape(b, c, -1)
    feature_2 = feature_2.reshape(b, c, -1)

    l = feature_1.shape[-1]
    dist = (torch.arange(l).view(l, 1) - torch.arange(l).view(1, l)).abs().float().cuda()
    D = dist / dist.max()
    D = D.cpu().numpy()
    _M = torch.from_numpy(D).float().unsqueeze(0).cuda()  # b, m, n

    result_dot = torch.tensor(sys.maxsize).cuda()
    for i in range(feature_1.shape[0]):  # Feature_1 batch循环
        for j in range(feature_2.shape[0]):  # Feature_2 batch循环
            result_tmp = 0
            sinkhorn_sum = 0
            l2_sum = 0
            # 计算整张图的sinkhorn distance和l2 distance
            for k in range(feature_1.shape[1]):  # 通道循环
                # 计算sinkhorn_distance
                P = sinkhorn_distance(feature_1[i][k].unsqueeze(0), feature_2[j][k].unsqueeze(0), _M, reg=1e-2)
                P = P.squeeze(0).detach().cpu().numpy()
                # 使用最小最大归一化，将特征图的值缩放到0和1之间
                min_val = P.min()  # 计算特征图的最小值
                max_val = P.max()  # 计算特征图的最大值
                P = (P - min_val) / (max_val - min_val)
                T = P * D
                sinkhorn_sum = np.sum(T)
                # 计算l2距离的平方
                squared_diff = (feature_1[i][k] - feature_2[j][k]) ** 2
                l2_sum = torch.sum(squared_diff)
                # 计算dot
                result_tmp += torch.sum(torch.sqrt(l2_sum) * sinkhorn_sum) + beta * sinkhorn_sum * torch.log(torch.tensor(sinkhorn_sum))
            # 保存最优的结果
            # 计算dot
            if result_tmp < result_dot:
                # 更新最好的结果，并保存index
                result_dot = result_tmp

    return result_dot

def get_dataset_kitti(dir_path_img, top):
    # 遍历目录下的所有文件
    index = 0
    #设置获取上限
    top_index = top
    device = torch.device("cuda")

    image_list = []
    image_list_path = []
    for filename in os.listdir(dir_path_img):
        input_image = pil.open(dir_path_img + filename).convert('RGB')
        input_image = input_image.resize(
            (1024, 1024), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)
        image_list.append(input_image)
        image_list_path.append(dir_path_img + filename)
        index += 1
        if index >= top_index:
            break

    return image_list, image_list_path

def get_dataset(dir_path_img, dir_path_gt):
    device = torch.device("cuda")
    # 遍历目录下的所有文件
    image_list = []
    image_list_path = []
    gt_list_path = []
    for filename in os.listdir(dir_path_img):
        input_image = pil.open(dir_path_img + filename).convert('RGB')
        input_image = input_image.resize(
            (1024, 1024), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(device)
        image_list.append(input_image)
        image_list_path.append(dir_path_img + filename)
        
    for filename in os.listdir(dir_path_gt):
        gt_list_path.append(dir_path_gt + filename)
    

    return image_list, image_list_path, gt_list_path

def get_img(path):
    input_image = pil.open(path).convert('RGB')
    input_image = transforms.ToTensor()(input_image)
    input_image = input_image
    return input_image

def write_text_to_file(file_number):
    # 格式化文件名为5位数字，前面补0
    file_name = f"{file_number:06d}"
    
    return file_name

def create_model(args):
    print("=================加载预训练权重中================")
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)
    print("================encoder加载预训练权重完毕！！！================")

    CVP_path = os.path.join(args.model_path, "CycledViewProjection.pth")
    CVP_dict = torch.load(CVP_path, map_location=device)
    models['CycledViewProjection'] = CycledViewProjection(in_dim=8)
    filtered_dict_cvp = {
        k: v for k,
        v in CVP_dict.items() if k in models["CycledViewProjection"].state_dict()}
    models["CycledViewProjection"].load_state_dict(filtered_dict_cvp)
    print("================CycledViewProjection加载预训练权重完毕！！！================")

    CVT_path = os.path.join(args.model_path, "CrossViewTransformer.pth")
    CVT_dict = torch.load(CVT_path, map_location=device)
    models['CrossViewTransformer'] = CrossViewTransformer(128)
    filtered_dict_cvt = {
        k: v for k,
        v in CVT_dict.items() if k in models["CrossViewTransformer"].state_dict()}
    models["CrossViewTransformer"].load_state_dict(filtered_dict_cvt)
    print("================CrossViewTransformer加载预训练权重完毕！！！================")

    decoder_path = os.path.join(args.model_path, "decoder.pth")
    DEC_dict = torch.load(decoder_path, map_location=device)
    models["decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc)
    filtered_dict_dec = {
        k: v for k,
        v in DEC_dict.items() if k in models["decoder"].state_dict()}
    models["decoder"].load_state_dict(filtered_dict_dec)
    print("================decoder加载预训练权重完毕！！！================")

    transform_decoder_path = os.path.join(args.model_path, "transform_decoder.pth")
    TRDEC_dict = torch.load(transform_decoder_path, map_location=device)
    models["transform_decoder"] = model.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc)
    filtered_dict_trdec = {
        k: v for k,
        v in TRDEC_dict.items() if k in models["transform_decoder"].state_dict()}
    models["transform_decoder"].load_state_dict(filtered_dict_trdec)
    print("================transform_decoder加载预训练权重完毕！！！================")

    for key in models.keys():
        models[key].to(device)
        models[key].eval()
    return models

def calculate_feature(models, feature):
    features = models["encoder"](feature)
    # transform_feature, retransform_features = models["CycledViewProjection"](features)

    # result = models["CrossViewTransformer"](features, transform_feature, retransform_features)

    return features

def create_feature(models, feature_1, feature_2):
    device = torch.device("cuda")
    # 移植到device
    feature_1 = feature_1.to(device)
    feature_2 = feature_2.to(device)

    # 计算特征向量
    feature_1 = calculate_feature(models, feature_1)
    feature_2 = calculate_feature(models, feature_2)

    return feature_1, feature_2

def convert_gt(path):
    Origin_Point_Value = np.array([0, 255])
    Out_Point_Value = np.array([0, 1])
    png  = Image.open(path)

    w, h    = png.size
    png     = np.array(png)
    out_png = np.zeros([h, w])
    for i in range(len(Origin_Point_Value)):
        mask = png[:, :] == Origin_Point_Value[i]
        if len(np.shape(mask)) > 2:
            mask = mask.all(-1)
        out_png[mask] = Out_Point_Value[i]
    
    return torch.from_numpy(out_png).cuda()

def carla_kitti_label(carla_path, kitti_path):
    carla = convert_gt(carla_path)
    kitti = convert_gt(kitti_path)
    is_binary_carla = torch.all(torch.eq(carla, 0) | torch.eq(carla, 1))
    is_binary_kitti = torch.all(torch.eq(kitti, 0) | torch.eq(kitti, 1))

    print(f"Carla Matrix contains only 0s and 1s: {is_binary_carla.item()}")
    print(f"Kitti Matrix contains only 0s and 1s: {is_binary_kitti.item()}")

    result_matrix = torch.where(carla != kitti, carla + kitti, carla)
    is_binary_kitti = torch.all(torch.eq(result_matrix, 0) | torch.eq(result_matrix, 1))
    print(f"Result Matrix contains only 0s and 1s: {is_binary_kitti.item()}")
    return result_matrix

def Retrieval_merge_neighbor(models, index, type, carla_dir_path_img, carla_dir_path_gt, kitti_dir_path_img, kitti_dir_path_gt, carla_ratio, carla_num, kitti_num):
    carla_dir_path_img = carla_dir_path_img
    carla_dir_path_gt = carla_dir_path_gt
    kitti_dir_path_img = kitti_dir_path_img
    kitti_dir_path_gt = kitti_dir_path_gt
    carla_image_list, carla_image_list_path, carla_gt_list_path = get_dataset_kitti(carla_dir_path_img,
            carla_dir_path_gt, carla_num)
    print("读取carla数据集完毕")
    kitti_image_list, kitti_image_list_path, kitti_gt_list_path = get_dataset_kitti(kitti_dir_path_img,
            kitti_dir_path_gt, kitti_num)
    print("读取kitti数据集完毕")

    beta = 1e-3

    # 可视化融合的特征和label
    save_img_path = './new_fine_turn/img/'
    save_img_rgb_path = './new_fine_turn/feat/'
    save_gt_path = './new_fine_turn/gt/'
    index = index


    print("检索并合并")    
    # 访问carla图片
    for carla_index, carla_img in enumerate(carla_image_list):
        carla_feature = calculate_feature(models, carla_img.cuda())
        result_dot = torch.tensor(sys.maxsize).cuda()
        path_index = 0
        kitti_save_feature = None
        # 访问kitti图片
        for kitti_index, kitti_img in enumerate(kitti_image_list):
            kitti_feature = calculate_feature(models, kitti_img.cuda())
            # 计算距离
            result = calculate_dot_feature_cuda(carla_feature, kitti_feature, beta)
            # print(result, result_dot)
            if result < result_dot:
                result_dot = result
                path_index = kitti_index
                kitti_save_feature = kitti_feature
                print("检索到更近的邻居carla_img_index{%d},结果为{%f}" % (path_index, result_dot))
        
        # 融合特征
        print("融合kitti_index{%d}, carla_index{%d}"  % (path_index, carla_index))
        if type == 'convex':
            merge_feature = carla_ratio * carla_feature + (1 - carla_ratio) * kitti_save_feature
        else:
            merge_feature = carla_feature + kitti_save_feature
        merge_gt = carla_kitti_label(carla_gt_list_path[carla_index],kitti_gt_list_path[path_index])
        # 保存融合特征和label标签
        data = (merge_feature.detach().to('cpu'), merge_gt.detach().to('cpu'))
        torch.save(data, "./new_fine_turn/pt/" + type + "/train/" + write_text_to_file(index) + ".pt")
        torch.save(data, "./new_fine_turn/pt/" + type + "/val/" + write_text_to_file(index) + ".pt")


        # #保存融合的图片信息，方便检查是否正确
        with open("./new_fine_turn/all.txt", 'a') as file:
            file.write(str(carla_image_list_path[carla_index]) + "=====")
            file.write(str(kitti_image_list_path[path_index]) + "\n")
            file.write(str(carla_gt_list_path[carla_index]) + "=====")
            file.write(str(kitti_gt_list_path[path_index]) + "\n")
        # to_pil = ToPILImage()
        # image_feature = to_pil((carla_img + kitti_image_list[path_index]).squeeze(0))
        # merge_gt = to_pil((get_img(kitti_gt_list_path[path_index]) + get_img(carla_gt_list_path[carla_index])))

        # # 将PIL图像保存为PNG文件
        # image_feature.save(save_img_path + write_text_to_file(index)+ '.png')
        # to_pil(carla_img.squeeze(0)).save(save_img_rgb_path + write_text_to_file(index)+ '.png')
        # to_pil(kitti_image_list[path_index].squeeze(0)).save(save_img_rgb_path + write_text_to_file(index)+ '_kitii.png')
        # merge_gt.save(save_gt_path + write_text_to_file(index) + '.png')
        # with open("./new_fine_turn/data_train.txt", 'a') as file:
        #     file.write(write_text_to_file(index) + "\n")
        index += 1
        print("Already create one epoch")

def Get_kitti_feature(models, index, kitti_dir_path_img, kitti_dir_path_gt, kitti_num):

    kitti_image_list, kitti_image_list_path, kitti_gt_list_path = get_dataset_kitti(kitti_dir_path_img,
            kitti_dir_path_gt, kitti_num)

    for kitti_index, kitti_img in enumerate(kitti_image_list):
        kitti_feature = calculate_feature(models, kitti_img.cuda())
        data_real = (kitti_feature.detach().to('cpu'), convert_gt(kitti_gt_list_path[kitti_index]).detach().to('cpu'))
        torch.save(data_real, "./new_fine_turn/pt/convex/train/" + write_text_to_file(index) + ".pt")
        index += 1

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

# 步骤2: 创建Dataset子类
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 假设每个.pt文件保存的是一个元组(feature_map, ground_truth)
        feature_map, ground_truth = self.data_list[idx]
        return feature_map, ground_truth

class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {"static": self.opt.static_weight, "dynamic": self.opt.dynamic_weight}
        self.seed = self.opt.global_seed
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.transform_parameters_to_train = []
        self.detection_parameters_to_train = []
        self.base_parameters_to_train = []
        self.parameters_to_train = []
        self.parameters_to_train_D = []
        self.criterion = compute_losses_fine_turn()
        self.create_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.epoch = 0
        self.start_epoch = 0
        self.scheduler = 0
        # Save log and models path
        self.opt.log_root = os.path.join(self.opt.log_root, self.opt.split)
        self.opt.save_path = os.path.join(self.opt.save_path, self.opt.split)
        if self.opt.split == "argo":
            self.opt.log_root = os.path.join(self.opt.log_root, self.opt.type)
            self.opt.save_path = os.path.join(self.opt.save_path, self.opt.type)
        self.writer = SummaryWriter(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time))
        self.log = open(os.path.join(self.opt.log_root, self.opt.model_name, self.create_time,
                                     '%s.csv' % self.opt.model_name), 'w')

        if self.seed != 0:
            self.set_seed()  # set seed

        # Initializing models
        self.models["encoder"] = crossView.Encoder(18, self.opt.height, self.opt.width, True)

        self.models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
        self.models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)

        self.models["decoder"] = crossView.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class)
        self.models["transform_decoder"] = crossView.Decoder(
            self.models["encoder"].resnet_encoder.num_ch_enc, self.opt.num_class, "transform_decoder")

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(
                    self.models[key].parameters())
            elif "transform" in key:
                self.transform_parameters_to_train += list(self.models[key].parameters())
            else:
                if key == 'decoder':
                    self.base_parameters_to_train += list(self.models[key].parameters())
        self.parameters_to_train = [
            {"params": self.transform_parameters_to_train, "lr": self.opt.lr_transform},
            {"params": self.base_parameters_to_train, "lr": self.opt.lr},
        ]

        # Optimization
        self.model_optimizer = optim.Adam(
            self.parameters_to_train)
        # self.scheduler = ExponentialLR(self.model_optimizer, gamma=0.98)
        # self.scheduler = StepLR(self.model_optimizer, step_size=step_size, gamma=0.65)
        self.scheduler = MultiStepLR(self.model_optimizer, milestones=self.opt.lr_steps, gamma=0.1)
        # self.scheduler = CosineAnnealingLR(self.model_optimizer, T_max=15)  # iou 35.55

        self.patch = (1, self.opt.occ_map_size // 2 **
                      4, self.opt.occ_map_size // 2 ** 4)

        self.valid = Variable(
            torch.Tensor(
                np.ones(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()
        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()

        # Data Loaders
        # 训练文件路径
        # train_filenames = './new_fine_turn/pt/simple_add/train'
        # val_filenames = './new_fine_turn/pt/simple_add/val'
        train_filenames = './new_fine_turn/pt/convex/train'
        val_filenames = './new_fine_turn/pt/convex/val'

        # 步骤1: 加载所有.pt文件
        train_data_list = []
        for pt_file in os.listdir(train_filenames):
            if pt_file.endswith('.pt'):
                data = torch.load(os.path.join(train_filenames, pt_file))
                train_data_list.append(data)
        
        val_data_list = []
        for pt_file in os.listdir(val_filenames):
            if pt_file.endswith('.pt'):
                data = torch.load(os.path.join(val_filenames, pt_file))
                val_data_list.append(data)

        train_dataset = MyDataset(train_data_list)
        val_dataset = MyDataset(val_data_list)

        self.train_dataloader = DataLoader(train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        if self.opt.load_weights_folder != "":
            print("加载预训练权重")
            self.load_model()

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):
        if not os.path.isdir(self.opt.log_root):
            os.mkdir(self.opt.log_root)

        for self.epoch in range(self.start_epoch, self.opt.num_epochs + 1):
            self.adjust_learning_rate(self.model_optimizer, self.epoch, self.opt.lr_steps)
            loss = self.run_epoch()
            output = ("Epoch: %d | lr:%.7f | Loss: %.4f | topview Loss: %.4f | transform_topview Loss: %.4f | "
                      "transform Loss: %.4f"
                      % (self.epoch, self.model_optimizer.param_groups[-1]['lr'], loss["loss"], loss["topview_loss"],
                         loss["transform_topview_loss"], loss["transform_loss"]))
            print(output)
            self.log.write(output + '\n')
            self.log.flush()
            for loss_name in loss:
                self.writer.add_scalar(loss_name, loss[loss_name], global_step=self.epoch)
            if self.epoch % self.opt.log_frequency == 0:
                self.validation(self.log)
                if self.opt.model_split_save:
                    self.save_model()
        self.save_model()

    def process_batch(self, features, labels = None, validation=False):
        outputs = {}
        # (6, 1, 128, 8, 8)
        features = torch.squeeze(features.float()).cuda()
        if validation:
            features = features.unsqueeze(0)
        
        # Cross-view Transformation Module
        x_feature = features
        transform_feature, retransform_features = self.models["CycledViewProjection"](features)
        features = self.models["CrossViewTransformer"](features, transform_feature, retransform_features)

        outputs["topview"] = self.models["decoder"](features)
        outputs["transform_topview"] = self.models["transform_decoder"](transform_feature)

        if validation:
            return outputs
        losses = self.criterion(self.opt, self.weight, labels, outputs, x_feature, retransform_features)
        # losses = self.criterion(self.opt, self.weight, labels, outputs)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        loss = {
            "loss": 0.0,
            "topview_loss": 0.0,
            "transform_loss": 0.0,
            "transform_topview_loss": 0.0,
            "loss_discr": 0.0
        }
        
        accumulation_steps = 8
        for feature, labels in self.train_dataloader:
            device = self.device
            feature = feature.to(device)
            labels = labels.to(device)
            outputs, losses = self.process_batch(feature, labels)
            self.model_optimizer.zero_grad()

            losses["loss"] = losses["loss"] / accumulation_steps
            losses["loss"].backward()
            self.model_optimizer.step()

            for loss_name in losses:
                loss[loss_name] += losses[loss_name].item()
        self.scheduler.step()
        for loss_name in loss:
            loss[loss_name] /= len(self.train_dataloader)

        return loss

    def validation(self, log):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
        trans_iou, trans_mAP = np.array([0., 0.]), np.array([0., 0.])
        for feature, labels in self.val_loader:
            with torch.no_grad():
                outputs = self.process_batch(feature, validation = True)
            # [6, 256, 256]
            pred = np.squeeze(
                torch.argmax(
                    outputs["topview"].detach(),
                    1).cpu().numpy())
            true = np.squeeze(
                labels.detach().cpu().numpy())
            iou += mean_IU(pred, true)
            mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        output = ("Epoch: %d | Validation: mIOU: %.2f %% mAP: %.2f %%" % (self.epoch, iou[1] * 100, mAP[1] * 100))
        print(output)
        log.write(output + '\n')
        log.flush()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            "weights_{}".format(
                self.epoch)
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            state_dict['epoch'] = self.epoch
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        for key in self.models.keys():
            if "discriminator" not in key:
                print("Loading {} weights...".format(key))
                path = os.path.join(
                    self.opt.load_weights_folder,
                    "{}.pth".format(key))
                model_dict = self.models[key].state_dict()
                pretrained_dict = torch.load(path)
                if 'epoch' in pretrained_dict:
                    self.start_epoch = pretrained_dict['epoch']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[key].load_state_dict(model_dict)

        # loading adam state
        if self.opt.load_weights_folder == "":
            optimizer_load_path = os.path.join(
                self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")

    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        decay = round(decay, 2)
        lr = self.opt.lr * decay
        lr_transform = self.opt.lr_transform * decay
        decay = self.opt.weight_decay
        optimizer.param_groups[0]['lr'] = lr_transform
        optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = decay
        optimizer.param_groups[1]['weight_decay'] = decay

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    

if __name__ == "__main__":
    # #检索
    # 加载预训练权重
    args = get_args()
    # 创建模型及特征图
    models = create_model(args)
    carla_1_dir_path_img = "./OT_pic/carla/"
    kitti_dir_path_img = "./OT_pic/kitti/"
    
    # 获取照片
    carla_image_list, carla_image_list_path = get_dataset_kitti(carla_1_dir_path_img, 4)
    print("读取carla数据集完毕")
    kitti_image_list, kitti_image_list_path = get_dataset_kitti(kitti_dir_path_img, 1)
    print("读取kitti数据集完毕")

    beta = 1e-3

    # 获取特征
    # 计算OT距离
    for carla_index, carla_img in enumerate(carla_image_list):
        print(carla_image_list_path[carla_index])
        carla_feature = calculate_feature(models, carla_img.cuda())
        result_dot = torch.tensor(sys.maxsize).cuda()
        path_index = 0
        kitti_save_feature = None
        # 访问kitti图片
        for kitti_index, kitti_img in enumerate(kitti_image_list):
            kitti_feature = calculate_feature(models, kitti_img.cuda())
            # 计算距离
            result = calculate_dot_feature_cuda(carla_feature, kitti_feature, beta)
            print(result)