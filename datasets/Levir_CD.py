import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F
import cv2
from skimage.draw import random_shapes
num_classes = 1
MEAN = np.array([123.675, 116.28, 103.53])
STD  = np.array([58.395, 57.12, 57.375])
root = '/YOUR_DATA_ROOT/'

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    #im = (im - MEAN) / STD
    im = im/255
    return im.astype(np.float32)

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap

def Index2Color(pred):
    pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    return pred

def sliding_crop_CD(imgs1, imgs2, labels, edges, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    crop_edges = []
    label_dims = len(labels[0].shape)
    for img1, img2, label, edge in zip(imgs1, imgs2, labels, edges):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            crop_edges.append(edge)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                    crop_edges.append(edge[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                    crop_edges.append(edge[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' %len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels, crop_edges

def rand_crop_CD(img1, img2, label, edge, size):
    # print(img.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        crop_im2 = img2[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        crop_edge = edge[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im1, crop_im2, crop_label, crop_edge

def rand_flip_CD(img1, img2, label, edge):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img1, img2, label, edge
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(edge, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(edge, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), edge[::-1, ::-1].copy()
        
def rand_resize(img1, img2, label, edge, ratio):
    r = random.randint(0,3)
    h = img1.shape[0]
    w = img1.shape[1]
    # showIMG(img.transpose((1, 2, 0)))
    img1 = cv2.resize(img1, (int(w*ratio[r]), int(h*ratio[r])), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (int(w*ratio[r]), int(h*ratio[r])), interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (int(w*ratio[r]), int(h*ratio[r])), interpolation=cv2.INTER_AREA)
    edge = cv2.resize(edge, (int(w*ratio[r]), int(h*ratio[r])), interpolation=cv2.INTER_AREA)
    return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), edge[::-1, ::-1].copy()
        
def rand_exchange_CD(img1, img2, label, edge):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.5:
        return img2[::-1, ::-1, :].copy(), img1[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), edge[::-1, ::-1].copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), edge[::-1, ::-1].copy()
        
def rand_regroup_CD(img1, img2, label, edge, group_size, p):
    #print(label.shape)
    n = random.choice(group_size)
    if np.random.rand() < p:
        seed = np.random.randint(1,1000000000)
        h = img1.shape[0]
        w = img1.shape[1]
        block_size = h // n
        b_img1 = [img1[i:i+block_size, j:j+block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
        np.random.seed(seed)
        np.random.shuffle(b_img1)
        regroup_img1 = np.vstack([np.hstack(b_img1[i:i+n]) for i in range(0, n*n, n)])
    
        b_img2 = [img2[i:i+block_size, j:j+block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
        np.random.seed(seed)
        np.random.shuffle(b_img2)
        regroup_img2 = np.vstack([np.hstack(b_img2[i:i+n]) for i in range(0, n*n, n)])
    
        b_label = [label[i:i+block_size, j:j+block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
        np.random.seed(seed)
        np.random.shuffle(b_label)
        regroup_label = np.vstack([np.hstack(b_label[i:i+n]) for i in range(0, n*n, n)])
        
        b_edge = [edge[i:i+block_size, j:j+block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
        np.random.seed(seed)
        np.random.shuffle(b_edge)
        regroup_edge = np.vstack([np.hstack(b_edge[i:i+n]) for i in range(0, n*n, n)])
        #print(regroup_label.shape)
        return regroup_img1, regroup_img2, regroup_label, regroup_edge
    else:
        return img1, img2, label, edge

def read_RSimages(mode, read_list=False):
    assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_dir = os.path.join(root, mode, 'label')
    
    if mode=='train' and read_list:
        list_path=os.path.join(root, mode+'0.4_info.txt')
        list_info = open(list_path, 'r')
        data_list = list_info.readlines()
        data_list = [item.rstrip() for item in data_list]
    else:
        data_list = os.listdir(img_A_dir)
    data_A, data_B, labels, edges = [], [], [], []
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_path = os.path.join(label_dir, it)
            
            img_A = io.imread(img_A_path)
            img_A = normalize_image(img_A)
            img_B = io.imread(img_B_path)
            img_B = normalize_image(img_B)
            label = Color2Index(io.imread(label_path))
            edge = Color2Index(cv2.Canny(cv2.imread(label_path), threshold1=100, threshold2=200)).astype(np.uint8) 
            data_A.append(img_A)
            data_B.append(img_B)
            labels.append(label)
            edges.append(edge)
        #if idx>10: break    
        if not idx%50: print('%d/%d images loaded.'%(idx, len(data_list)))
    print(data_A[0].shape)
    print(str(len(data_A)) + ' ' + mode + ' images loaded.')   
    return data_A, data_B, labels, edges
    
class GaussianNoise(object):
    def __init__(self, std=0.05):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label]
def random_mask(img_a, img_b, max_shapes=5, min_size=20):
    """
    在图片A上随机生成掩码并融合图片B对应区域（修复版）
    """
    # 读取图片（保持RGB格式）
    #img_a = iio.imread(img_a_path).astype(np.uint8)
    #img_b = iio.imread(img_b_path).astype(np.uint8)
    
    # 验证图片尺寸一致
    if img_a.shape[:2] != img_b.shape[:2]:
        raise ValueError("输入图片尺寸必须相同！")
    
    # 生成随机形状掩码
    mask_shape, _ = random_shapes(
        img_a.shape[:2],
        max_shapes=max_shapes,
        min_shapes=1,
        min_size=min_size,
        max_size=min(img_a.shape[:2])//2,
        #intensity_range=(0, 1),
        allow_overlap=False,
        channel_axis=None,
        #shape='ellipse' if np.random.rand()>0.5 else 'rectangle'
    )
    
    # 创建融合图片
    fused_img = img_a.copy()
    mask = (mask_shape == 255).astype(np.uint8)
    # 提取掩码区域（将mask转换为布尔类型）
    mask_bool = mask.astype(bool)
    # 修复后的区域有效性验证（三维操作）
    # 检查B图像对应区域是否存在有效像素（非全黑）
    b_valid_regions = (img_b.sum(axis=-1) != 0)  # 保持(H,W)形状
    # 计算最终有效掩码（保持二维布尔运算）
    valid_mask = mask_bool & b_valid_regions
    
    # 执行融合操作
    fused_img[mask_bool] = img_b[mask_bool]
    
    # 保存二进制掩码（转换为0-1格式）
    binary_mask = 1 - mask_bool.astype(np.uint8) 
    edge = Color2Index(cv2.Canny(binary_mask, threshold1=100, threshold2=200))
    
    return img_b, fused_img, binary_mask, edge
    
def process_images_with_mask(img_A, img_B, mask_C, num_regions=20):
    """

    处理图像和掩码：
    1. 从A和B中提取掩码区域
    2. 随机选择部分区域
    3. 随机粘贴到原图任意位置

    4. 生成新掩码
    
    参数:
        img_A, img_B: 输入图像(H,W,3)

        mask_C: 二进制掩码(H,W)
        num_regions: 要随机选择的区域数量
        
    返回:
        new_A, new_B: 处理后的图像
        new_mask: 合并后的新掩码
    """
    assert img_A.shape[:2] == img_B.shape[:2] == mask_C.shape
    
    # 复制原始数据
    new_A = img_A.copy()
    new_B = img_B.copy()
    new_mask = mask_C.copy()
    
    # 查找连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_C.astype(np.uint8))
    
    # 过滤有效区域(忽略背景和太小的区域)
    valid_regions = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 50]
    if len(valid_regions) > 1:
        num_regions = random.randint(1, len(valid_regions))
        selected_regions = random.sample(valid_regions, num_regions)
    else:
        selected_regions = valid_regions        
    # 随机选择指定数量的区域
    #if len(valid_regions) > num_regions:
    #    selected_regions = random.sample(valid_regions, num_regions)
    #else:
    #    selected_regions = valid_regions
    
    for region in selected_regions:
        # 获取区域边界框
        x, y, w, h, _ = stats[region]
        
        # 提取区域图像和掩码
        region_A = img_A[y:y+h, x:x+w]
        region_B = img_B[y:y+h, x:x+w]
        region_mask = (labels[y:y+h, x:x+w] == region).astype(np.uint8)
        
        # 随机缩放因子(0.5-1.5倍)
        scale = random.uniform(0.5, 1.5)
        new_h, new_w = int(h*scale), int(w*scale)
        
        # 随机旋转角度(-45到45度)
        angle = random.uniform(-45, 45)
        
        # 计算旋转中心
        center = (new_w//2, new_h//2)
        
        # 缩放和旋转图像区域
        M_scale = cv2.getRotationMatrix2D(center, 0, scale)
        M_rotate = cv2.getRotationMatrix2D(center, angle, 1)
        
        # 变换图像和掩码
        region_A = cv2.warpAffine(region_A, M_scale, (new_w, new_h))
        region_A = cv2.warpAffine(region_A, M_rotate, (new_w, new_h))
        
        region_B = cv2.warpAffine(region_B, M_scale, (new_w, new_h))
        region_B = cv2.warpAffine(region_B, M_rotate, (new_w, new_h))
        #if random.random() < 0.5:
        #    region_A, region_B = region_B, region_A
        
        region_mask = cv2.warpAffine(region_mask, M_scale, (new_w, new_h), flags=cv2.INTER_NEAREST)
        region_mask = cv2.warpAffine(region_mask, M_rotate, (new_w, new_h), flags=cv2.INTER_NEAREST)
        region_mask = (region_mask > 0).astype(np.uint8)
        
        # 随机选择粘贴位置(确保不超出边界)
        max_y = img_A.shape[0] - new_h
        max_x = img_A.shape[1] - new_w
        if max_y <= 0 or max_x <= 0:
            continue
            
        paste_y = random.randint(0, max_y)
        paste_x = random.randint(0, max_x)
        
        # 粘贴到新图像
        for c in range(3):
            new_A[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] = \
                new_A[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] * (1 - region_mask) + \
                region_A[:, :, c] * region_mask
            
            new_B[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] = \
                new_B[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] * (1 - region_mask) + \
                region_B[:, :, c] * region_mask
        
        # 更新掩码
        new_mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = np.maximum(
            new_mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w],
            region_mask)
    
    #binary_mask = 1 - new_mask.astype(np.uint8) 
    edge = Color2Index(cv2.Canny(new_mask, threshold1=100, threshold2=200)).astype(np.uint8) 
    return new_A, new_B, new_mask, edge

class RS(data.Dataset):
    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, random_regroup=False, random_mask=False, p=0.5, group_size=[4,8,16,32], crop_size=512, random_flip=False, rand_exchange_CD=False):
        self.random_flip = random_flip
        self.rand_exchange_CD = rand_exchange_CD
        self.random_crop = random_crop
        self.random_mask = random_mask
        self.random_regroup = random_regroup
        self.image_ratio = [ 0.75, 1, 1.25, 1.5]
        self.group_size = group_size
        self.image_resize = False
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        self.p = p
        data_A, data_B, labels, edges = read_RSimages(mode, read_list=False)
        self.dataset_len = len(data_A)-1
        if sliding_crop:
            data_A, data_B, labels, edges = sliding_crop_CD(data_A, data_B, labels, edges, [self.crop_size, self.crop_size])   
        self.data_A, self.data_B, self.labels, self.edges = data_A, data_B, labels, edges
        if self.random_crop:
            self.len = crop_nums*len(self.data_A)
        else:
            self.len = len(self.data_A)

    def __getitem__(self, idx):
        
        if self.random_crop:
            idx = idx//self.crop_nums
        data_A = self.data_A[idx]
        data_B = self.data_B[idx]
        label = self.labels[idx]
        edge = self.edges[idx]
        
        if self.random_mask and random.random()<0.5:
               data_A, data_B, label, edge = process_images_with_mask(data_A, data_B, label)  
               
        if self.random_regroup:
            data_A, data_B, label, edge = rand_regroup_CD(data_A, data_B, label, edge, self.group_size, self.p)
   
        if self.random_crop:
            if self.image_resize:
                 data_A, data_B, label, edge = rand_resize(data_A, data_B, label, edge, self.image_ratio) 
            data_A, data_B, label, edge = rand_crop_CD(data_A, data_B, label, edge, [self.crop_size, self.crop_size])
        if self.random_flip:
            data_A, data_B, label, edge = rand_flip_CD(data_A, data_B, label, edge)
        if self.rand_exchange_CD:
            data_A, data_B, label, edge = rand_exchange_CD(data_A, data_B, label, edge)  
  
        return F.to_tensor(data_A), F.to_tensor(data_B), label, edge

    def __len__(self):
        return self.len



