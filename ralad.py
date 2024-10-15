import numpy as np
import torch
import torch.nn as nn
import sys
from torchvision import transforms
import monolayout
import os
from PIL import Image
import PIL.Image as pil
from torchvision.transforms import ToPILImage
import argparse
from easydict import EasyDict as edict
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from utils import mean_IU, mean_precision
import random
import glob
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="./ckpt",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="monolayout",
                        help="Model Name with specifications")
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
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
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
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-4,
                        help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                        help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                        help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                        help="dynamic weight for calculating loss")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                        help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                        help="epoch to start training discriminator")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")

    return parser.parse_args()


class RALDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 假设每个.pt文件保存的是一个元组(feature_map, ground_truth)
        feature_map, ground_truth = self.data_list[idx]
        return feature_map, ground_truth


class RALAD:
    def run(self):
        args = get_args()
        # load weight
        # create model
        models = self.create_model(args)
        # txt path
        carla_txt_path = './splits/3Dobject/train_files_carla.txt'
        kitti_txt_path = './splits/3Dobject/train_files_kitti.txt'
        carla_txt_path_x = './splits/3Dobject/train_files_carla.txt'
        kitti_txt_path_x = './splits/3Dobject/train_files_kitti.txt'
        # image path
        dir_path_img = './fusion_dataset'

        # in-domain and out-domain
        path = "./new_fine_tune/pt/" + 'convex' + "/train/"
        self.ensure_directory_exists(path)
        num = self.count_files_in_directory(path)
        index = 0
        index += num
        self.Retrieval_merge_neighbor(models, index,'convex', carla_ratio=0.4, 
                                      carla_txt_path=carla_txt_path, kitti_txt_path=kitti_txt_path, 
                                      dir_path_img=dir_path_img, domain='cross')
        num = self.count_files_in_directory(path)
        index += num
        # out-domain
        self.Retrieval_merge_neighbor(models, index,'convex', carla_ratio=0.4, 
                                      carla_txt_path=carla_txt_path, kitti_txt_path=carla_txt_path_x, 
                                      dir_path_img=dir_path_img, domain='out')
        num = self.count_files_in_directory(path)
        index += num
        # in-domain
        self.Retrieval_merge_neighbor(models, index,'convex',carla_ratio=0.4, 
                                      carla_txt_path=kitti_txt_path, kitti_txt_path=kitti_txt_path_x, 
                                      dir_path_img=dir_path_img, domain='in')
    # To get the number of files in the directory, we can use the os.listdir method and count the files.

    def count_files_in_directory(self, directory_path):
        try:
            # List all files and directories in the given path
            files_and_dirs = os.listdir(directory_path)
            # Filter out directories and count files
            file_count = sum(1 for item in files_and_dirs if os.path.isfile(os.path.join(directory_path, item)))
            return file_count
        except Exception as e:
            return f"Error: {e}"
        
    
    def sinkhorn_distance(self, r, c, M, reg=1e-2, error_thres=1e-5, niter=100):
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

    def calculate_dot_feature(self, feature_1, feature_2, beta):
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
                    P = self.sinkhorn_distance(feature_1[i][k].unsqueeze(0), feature_2[j][k].unsqueeze(0), _M,reg=1e-2)
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

    def calculate_dot(self, feature_1, feature_2, beta):
        l = feature_1.shape[-1]
        dist = (torch.arange(l).view(l, 1) - torch.arange(l).view(1, l)).abs().float()
        D = dist / dist.max()
        D = D.cpu().numpy()
        _M = torch.from_numpy(D).float().unsqueeze(0)  # b, m, n

        # 计算sinkhorn_distance
        P = self.sinkhorn_distance(feature_1, feature_2, _M,reg=1e-2)
        P = P.squeeze(0).detach().cpu().numpy()
        T = P * D
        sinkhorn_sum = np.sum(T)
        # 计算l2距离的平方
        squared_diff = (feature_1 - feature_2) ** 2
        l2_sum = torch.sum(squared_diff)

        # 计算dot
        return torch.sum(l2_sum * sinkhorn_sum) + beta * sinkhorn_sum * torch.log(torch.tensor(sinkhorn_sum))

    def calculate_dot_feature_cuda(self, feature_1, feature_2, beta):
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
                    P = self.sinkhorn_distance(feature_1[i][k].unsqueeze(0), feature_2[j][k].unsqueeze(0), _M, reg=1e-2)
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

    def get_dataset_kitti(self, dir_path_img, dir_path_gt, top):
        # 遍历目录下的所有文件
        index = 0
        #设置获取上限
        top_index = top
        device = torch.device("cuda")

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
            index += 1
            if index >= top_index:
                break
        index = 0

        for filename in os.listdir(dir_path_gt):
            gt_list_path.append(dir_path_gt + filename)
            index += 1
            if index >= top_index:
                break

        return image_list, image_list_path, gt_list_path

    def get_dataset(self, dir_path_img, dir_path_gt):
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

    def get_img(self, path):
        input_image = pil.open(path).convert('RGB')
        input_image = transforms.ToTensor()(input_image)
        input_image = input_image
        return input_image

    def write_text_to_file(self, file_number):
        # 格式化文件名为5位数字，前面补0
        file_name = f"{file_number:06d}"
        
        return file_name

    def create_model(self, args):
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

    def calculate_feature(self, models, feature):
        features = models["encoder"](feature)

        return features

    def create_feature(self, models, feature_1, feature_2):
        device = torch.device("cuda")
        # 移植到device
        feature_1 = feature_1.to(device)
        feature_2 = feature_2.to(device)

        # 计算特征向量
        feature_1 = self.calculate_feature(models, feature_1)
        feature_2 = self.calculate_feature(models, feature_2)

        return feature_1, feature_2

    def carla_kitti_label(self, carla_path, kitti_path):
        print(carla_path)
        carla = self.convert_gt(carla_path)
        kitti = self.convert_gt(kitti_path)
        is_binary_carla = torch.all(torch.eq(carla, 0) | torch.eq(carla, 1))
        is_binary_kitti = torch.all(torch.eq(kitti, 0) | torch.eq(kitti, 1))

        print(f"Carla Matrix contains only 0s and 1s: {is_binary_carla.item()}")
        print(f"Kitti Matrix contains only 0s and 1s: {is_binary_kitti.item()}")

        result_matrix = torch.where(carla != kitti, carla + kitti, carla)
        is_binary_kitti = torch.all(torch.eq(result_matrix, 0) | torch.eq(result_matrix, 1))
        print(f"Result Matrix contains only 0s and 1s: {is_binary_kitti.item()}")
        return result_matrix

    def get_img_OT(self, dir_path_img, carla_name):
        carla_img = pil.open(dir_path_img + '/' + carla_name + '.png').convert('RGB')
        carla_img = carla_img.resize(
            (1024, 1024), pil.LANCZOS)
        carla_img = transforms.ToTensor()(carla_img).unsqueeze(0)
        carla_img = carla_img.to('cuda')
        return carla_img

    def ensure_directory_exists(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return f"Directory created: {directory_path}"
        else:
            return f"Directory already exists: {directory_path}"

    def Retrieval_merge_neighbor(self, models, index, type, carla_ratio, carla_txt_path, kitti_txt_path, dir_path_img, domain):
            carla_img_list = self.read_picture_names_from_file(carla_txt_path)
            kitti_img_list = self.read_picture_names_from_file(kitti_txt_path)
            print("KITTI %d images and CARLA %d images" % (len(kitti_img_list ),len(carla_img_list)))
            beta = 1e-3
            index = index

            print("Retrieve and merge")    
            # Visit Carla Images
            if domain == 'cross':
                for carla_index, carla_name in enumerate(carla_img_list):
                    # get image
                    carla_img = self.get_img_OT(dir_path_img + '/carla/image_2', carla_name)
                    # get feature
                    carla_feature = self.calculate_feature(models, carla_img.cuda())
                    result_dot = torch.tensor(sys.maxsize).cuda()
                    path_index = 0
                    kitti_save_feature = None
                    # Visit Kitti Images
                    for kitti_index, kitti_name in enumerate(kitti_img_list):
                        # get image
                        kitti_img = self.get_img_OT(dir_path_img + '/kitti/image_2', kitti_name)
                        kitti_feature = self.calculate_feature(models, kitti_img.cuda())
                        # Calculate distance
                        result = self.calculate_dot_feature_cuda(carla_feature, kitti_feature, beta)
                        if result < result_dot:
                            result_dot = result
                            path_index = kitti_index
                            kitti_save_feature = kitti_feature
                            print("Retrieve closer neighbors carla_img_index{%d}, the result is{%f}" % (path_index, result_dot))
                    
                    # Fusion features
                    print("Merge kitti_index{%d}, carla_index{%d}"  % (path_index, carla_index))
                    if type == 'convex':
                        merge_feature = carla_ratio * carla_feature + (1 - carla_ratio) * kitti_save_feature
                    else:
                        merge_feature = carla_feature + kitti_save_feature
                    
                    kitti_gt_path = dir_path_img + '/kitti/vehicle_256/' + kitti_img_list[path_index] +'.png'
                    carla_gt_path = dir_path_img + '/carla/vehicle_256/' + carla_name +'.png'
                    merge_gt = self.carla_kitti_label(carla_gt_path, kitti_gt_path)
                    # Save fusion features and label labels
                    data = (merge_feature.detach().to('cpu'), merge_gt.detach().to('cpu'))
                    save_pt_path = "./new_fine_tune/pt/" + type + "/train/"
                    self.ensure_directory_exists(save_pt_path)
                    torch.save(data, save_pt_path + self.write_text_to_file(index) + ".pt")

                    index += 1
                    print("Already create one epoch")
            elif domain == 'out':
                for carla_index, carla_name in enumerate(carla_img_list):
                    # get image
                    carla_img = self.get_img_OT(dir_path_img + '/carla/image_2', carla_name)
                    # get feature
                    carla_feature = self.calculate_feature(models, carla_img.cuda())
                    result_dot = torch.tensor(sys.maxsize).cuda()
                    path_index = 0
                    kitti_save_feature = None
                    # Visit Kitti Images
                    for kitti_index, kitti_name in enumerate(kitti_img_list):
                        # get image
                        kitti_img = self.get_img_OT(dir_path_img + '/carla_out/image_2', kitti_name)
                        kitti_feature = self.calculate_feature(models, kitti_img.cuda())
                        # Calculate distance
                        result = self.calculate_dot_feature_cuda(carla_feature, kitti_feature, beta)
                        if result < result_dot:
                            result_dot = result
                            path_index = kitti_index
                            kitti_save_feature = kitti_feature
                            print("Retrieve closer neighbors carla_out_img_index{%d}, the result is{%f}" % (path_index, result_dot))
                    
                    # Fusion features
                    print("Merge carla_out_index{%d}, carla_index{%d}"  % (path_index, carla_index))
                    if type == 'convex':
                        merge_feature = carla_ratio * carla_feature + (1 - carla_ratio) * kitti_save_feature
                    else:
                        merge_feature = carla_feature + kitti_save_feature
                    
                    kitti_gt_path = dir_path_img + '/carla_out/vehicle_256/' + kitti_img_list[path_index] +'.png'
                    carla_gt_path = dir_path_img + '/carla/vehicle_256/' + carla_name +'.png'
                    merge_gt = self.carla_kitti_label(carla_gt_path, kitti_gt_path)
                    # Save fusion features and label labels
                    data = (merge_feature.detach().to('cpu'), merge_gt.detach().to('cpu'))
                    save_pt_path = "./new_fine_tune/pt/" + type + "/train/"
                    self.ensure_directory_exists(save_pt_path)
                    torch.save(data, save_pt_path + self.write_text_to_file(index) + ".pt")

                    index += 1
                    print("Already create one epoch")
            else:
                for carla_index, carla_name in enumerate(carla_img_list):
                    # get image
                    carla_img = self.get_img_OT(dir_path_img + '/kitti/image_2', carla_name)
                    # get feature
                    carla_feature = self.calculate_feature(models, carla_img.cuda())
                    result_dot = torch.tensor(sys.maxsize).cuda()
                    path_index = 0
                    kitti_save_feature = None
                    # Visit Kitti Images
                    for kitti_index, kitti_name in enumerate(kitti_img_list):
                        # get image
                        kitti_img = self.get_img_OT(dir_path_img + '/kitti_out/image_2', kitti_name)
                        kitti_feature = self.calculate_feature(models, kitti_img.cuda())
                        # Calculate distance
                        result = self.calculate_dot_feature_cuda(carla_feature, kitti_feature, beta)
                        if result < result_dot:
                            result_dot = result
                            path_index = kitti_index
                            kitti_save_feature = kitti_feature
                            print("Retrieve closer neighbors carla_img_index{%d}, the result is{%f}" % (path_index, result_dot))
                    
                    # Fusion features
                    print("Merge kitti_out_index{%d}, kitti_index{%d}"  % (path_index, carla_index))
                    if type == 'convex':
                        merge_feature = carla_ratio * carla_feature + (1 - carla_ratio) * kitti_save_feature
                    else:
                        merge_feature = carla_feature + kitti_save_feature
                    
                    kitti_gt_path = dir_path_img + '/kitti_out/vehicle_256/' + kitti_img_list[path_index] +'.png'
                    carla_gt_path = dir_path_img + '/carla/vehicle_256/' + carla_name +'.png'
                    merge_gt = self.carla_kitti_label(carla_gt_path, kitti_gt_path)
                    # Save fusion features and label labels
                    data = (merge_feature.detach().to('cpu'), merge_gt.detach().to('cpu'))
                    save_pt_path = "./new_fine_tune/pt/" + type + "/train/"
                    self.ensure_directory_exists(save_pt_path)
                    torch.save(data, save_pt_path + self.write_text_to_file(index) + ".pt")

                    index += 1
                    print("Already create one epoch")
    def Get_kitti_feature(self, models, index, kitti_dir_path_img, kitti_dir_path_gt, kitti_num):

        kitti_image_list, kitti_image_list_path, kitti_gt_list_path = self.get_dataset_kitti(kitti_dir_path_img,
                kitti_dir_path_gt, kitti_num)

        for kitti_index, kitti_img in enumerate(kitti_image_list):
            kitti_feature = self.calculate_feature(models, kitti_img.cuda())
            data_real = (kitti_feature.detach().to('cpu'), self.convert_gt(kitti_gt_list_path[kitti_index]).detach().to('cpu'))
            torch.save(data_real, "./new_fine_turn/pt/convex/train/" + self.write_text_to_file(index) + ".pt")
            index += 1

    def readlines(filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines

    def read_picture_names_from_file(self, file_path):
        with open(file_path, 'r') as file:
            # Read all lines and strip any leading/trailing whitespace
            names = [line.strip() for line in file.readlines()]
        return names

class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {}
        self.weight["static"] = self.opt.static_weight
        self.weight["dynamic"] = self.opt.dynamic_weight
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []

        # Initializing models
        self.models["encoder"] = monolayout.Encoder(
            18, self.opt.height, self.opt.width, True)
        if self.opt.type == "both":
            self.models["static_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["static_discr"] = monolayout.Discriminator()
            self.models["dynamic_decoder"] = monolayout.Discriminator()
            self.models["dynamic_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
        else:
            self.models["decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["discriminator"] = monolayout.Discriminator()

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(
                    self.models[key].parameters())
            else:
                self.parameters_to_train += list(self.models[key].parameters())

        # Optimization
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_D = optim.Adam(
            self.parameters_to_train_D, self.opt.lr)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.model_optimizer_D, self.opt.scheduler_step_size, 0.1)

        self.patch = (1, self.opt.occ_map_size // 2 **
                      4, self.opt.occ_map_size // 2**4)

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

        train_dataset = RALDataset(train_data_list)
        val_dataset = RALDataset(val_data_list)

        self.train_loader = DataLoader(train_dataset,
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
            self.load_model()

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):
        for self.epoch in range(self.opt.num_epochs):
            loss = self.run_epoch()
            print("Epoch: %d | Discriminator Loss: %.4f" %
                  (self.epoch, loss))

            if self.epoch % self.opt.log_frequency == 0:
                self.validation()
                self.save_model()

    def process_batch(self, features, labels = None, validation=False):
        outputs = {}
        features = torch.squeeze(features.float()).cuda()
        if validation:
            features = features.unsqueeze(0)

        if self.opt.type == "both":
            outputs["dynamic"] = self.models["dynamic_decoder"](features)
            outputs["static"] = self.models["static_decoder"](features)
        else:
            outputs["topview"] = self.models["decoder"](features)
        if validation:
            return outputs
        losses = self.compute_losses(labels, outputs)
        # losses["loss_discr"] = torch.zeros(1)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        self.model_optimizer_D.step()
        loss = {}
        loss["loss_discr"] = 0.0
        accumulation_steps = 8
        for feature, labels in self.train_loader:
            device = self.device
            feature = feature.to(device)
            labels = labels.to(device)
            outputs, losses = self.process_batch(feature, labels)
            self.model_optimizer.zero_grad()
            loss["loss_discr"] += losses["loss_discr"]
            losses["loss_discr"].backward()
            self.model_optimizer.step()
        for loss_name in loss:
            loss[loss_name] /= len(self.train_loader)

        return loss["loss_discr"]

    def validation(self):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
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
        print(
            "Epoch: %d | Validation: mIOU: %.4f mAP: %.4f" %
            (self.epoch, iou[1], mAP[1]))

    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.opt.type == "both":
            losses["static_loss"] = self.compute_topview_loss(
                                            outputs["static"],
                                            inputs,
                                            self.weight[self.opt.type])
            losses["dynamic_loss"] = self.compute_topview_loss(
                                            outputs["dynamic_loss"],
                                            inputs,
                                            self.weight[self.opt.type])
        else:
            losses["loss_discr"] = self.compute_topview_loss(
                                            outputs["topview"],
                                            inputs,
                                            self.weight[self.opt.type])

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):

        generated_top_view = outputs
        # true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view.long())
        return output.mean()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            self.opt.split,
            "weights_{}".format(
                self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
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
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.opt.load_weights_folder,
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

if __name__ == "__main__":
    # 微调
    start_time = time.ctime()
    print(start_time)
    trainer = Trainer()
    trainer.train()
    end_time = time.ctime()
    print(end_time)

    
