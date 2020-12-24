import os
import time
import math
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torchvision
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

################################################################################
################################################################################
################################################################################

batch_size = 16
learning_rate = 0.01
momentum = 0.1

hst = torch.device("cpu")
dev = torch.device("cuda")

grid_num = 7
grid_num_sq = grid_num*grid_num
bndbox_num = 2

idx_to_cls = [
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

cls_num = len(idx_to_cls)
cls_to_idx = {}
cls_cmp_tensor = torch.zeros(cls_num, cls_num)

for i in range(cls_num):
	cls_to_idx[idx_to_cls[i]] = i
	cls_cmp_tensor[i, i] = 1

data_num_per_grid = bndbox_num*5+cls_num

iou_th = 0.5
conf_th = 0.5
lambda_coord = 5  # = 5 in paper
lambda_noobj = 0.5  # = 0.5 in paper

################################################################################

std_size = (448, 448)

################################################################################
################################################################################
################################################################################

def TransferLabel(src_path, dst_path):
	src = ET.ElementTree(file = src_path).getroot()

	dst = ET.Element("label")
	dst.text = "\n"

	dst_size = ET.SubElement(dst, "size")
	dst_size.tail = "\n"

	src_size = src.find("size")
	size = [int(src_size.find("width").text),
			int(src_size.find("height").text)]

	ET.SubElement(dst_size, "width").text = str(std_size[0])
	ET.SubElement(dst_size, "height").text = str(std_size[1])

	for i in src.findall("object"):
		i_bndbox = i.find("bndbox")

		dst_bndbox = ET.SubElement(dst, "bndbox")
		dst_bndbox.tail = "\n"

		dst_bndbox_cls = ET.SubElement(dst_bndbox, "cls")
		dst_bndbox_cls.text = i.find("name").text

		a = [int(i_bndbox.find("xmin").text)*std_size[0]/size[0],
			 int(i_bndbox.find("ymin").text)*std_size[1]/size[1]]
		b = [int(i_bndbox.find("xmax").text)*std_size[0]/size[0],
			 int(i_bndbox.find("ymax").text)*std_size[1]/size[1]]

		ET.SubElement(dst_bndbox, "x").text = str(int((a[0]+b[0])/2))
		ET.SubElement(dst_bndbox, "y").text = str(int((a[1]+b[1])/2))
		ET.SubElement(dst_bndbox, "w").text = str(int(b[0]-a[0]))
		ET.SubElement(dst_bndbox, "h").text = str(int(b[1]-a[1]))

	ET.ElementTree(dst).write(dst_path)

def TransferImage(src_path, dst_path):
	Image.open(src_path).resize(std_size).save(dst_path)

def IoU_two_point(a_min, a_max, b_min, b_max):
	w = float(min(a_max[0], b_max[0])-max(a_min[0], b_min[0]))
	h = float(min(a_max[1], b_max[1])-max(a_min[1], b_min[1]))

	if (w <= 0) | (h <= 0):
		return 0

	cross = w*h
	return cross/(float(
		(a_max[0]-a_min[0])*(a_max[1]-a_min[1]) +
		(b_max[0]-b_min[0])*(b_max[1]-b_min[1]))-cross)

def IoU_two_xywh(a, b):
	w = float(min(a[0]+a[2]/2, b[0]+b[2]/2)-max(a[0]-a[2]/2, b[0]-b[2]/2))
	h = float(min(a[1]+a[3]/2, b[1]+b[3]/2)-max(a[1]-a[3]/2, b[1]-b[3]/2))

	if (w <= 0) | (h <= 0):
		return 0

	cross = w*h
	return cross/(float(a[2]*a[3]+b[2]*b[3])-cross)

################################################################################
################################################################################
################################################################################

def LoadImage(path):
	img = Image.open(path)
	data = torch.tensor(img.getdata(), dtype = torch.float).reshape(img.size[1], img.size[0], 3)  # h, w, c
	data = data.transpose(0, 2)  # c, w, h

	return data

def LoadLabel(path):
	gt = torch.zeros(grid_num, grid_num, bndbox_num, 1+cls_num+4)

	root = ET.ElementTree(file = path).getroot()
	root_size = root.find("size")
	size = [int(root_size.find("width").text),
			int(root_size.find("height").text)]

################################################################################

	obj_num_in_grid = torch.zeros(grid_num, grid_num, dtype = torch.uint8)

	for bndbox in root.findall("bndbox"):
		bndbox_pos = [int(bndbox.find("x").text), int(bndbox.find("y").text)]
		bndbox_size = [int(bndbox.find("w").text), int(bndbox.find("h").text)]

		grid = [
			int(bndbox_pos[0]*grid_num/size[0]),
			int(bndbox_pos[1]*grid_num/size[1])]

		obj_idx = int(obj_num_in_grid[grid[0]][grid[1]])

		if obj_idx == bndbox_num:
			continue

		gt[grid[0]][grid[1]][obj_idx][0] = 1  # set conf

		gt[grid[0]][grid[1]][obj_idx][1+cls_to_idx[bndbox.find("cls").text]] = 1  # set cls

		grid_pos = [
			int(size[0]*grid[0]/grid_num),
			int(size[1]*grid[1]/grid_num)]
		grid_size = [
			int(size[0]*(grid[0]+1)/grid_num)-grid_pos[0],
			int(size[1]*(grid[1]+1)/grid_num)-grid_pos[1]]

		gt[grid[0]][grid[1]][obj_idx][1+cls_num+0] = (bndbox_pos[0]-grid_pos[0])/grid_size[0]
		gt[grid[0]][grid[1]][obj_idx][1+cls_num+1] = (bndbox_pos[1]-grid_pos[1])/grid_size[1]
		gt[grid[0]][grid[1]][obj_idx][1+cls_num+2] = bndbox_size[0]/size[0]
		gt[grid[0]][grid[1]][obj_idx][1+cls_num+3] = bndbox_size[1]/size[1]
		# set xywh

		obj_num_in_grid[grid[0]][grid[1]] += 1

	return torch.flatten(gt)

################################################################################
################################################################################
################################################################################

def TransferLabelToConfClsXYWH_(label):
	# label	  : [batch_zie][grid*grid*bandox_num*(1+cls_num+4)]
	# label_conf : [batch_size*grid*grid][bndbox_num][1]
	# label_cls  : [batch_size*grid*grid][bndbox_num][cls_num]
	# label_xywh : [batch_size*grid*grid][bndbox_num][4]

	label_conf, label_cls, label_xywh = torch.split(
		torch.reshape(
			label, [-1, bndbox_num, 1+cls_num+4]
		), [1, cls_num, 4], 2)

	return label_conf, label_cls, label_xywh

def TransferLabelToConfClsXYWH(label):
	# label	  : [batch_zie][grid*grid*bandox_num*(1+cls_num+4)]
	# label_conf : [batch_size*grid*grid][bndbox_num]
	# label_cls  : [batch_size*grid*grid][bndbox_num][cls_num]
	# label_xywh : [batch_size*grid*grid][bndbox_num][4]

	label_conf, label_cls, label_xywh = TransferLabelToConfClsXYWH_(label)
	label_conf = torch.squeeze(label_conf)

	return label_conf, label_cls, label_xywh

################################################################################
################################################################################
################################################################################

def LabelImage(label, src_path, dst_path):
	img = Image.open(src_path)
	size = img.size
	img_draw = ImageDraw.Draw(img)

	label = torch.reshape(label, [grid_num, grid_num, bndbox_num*(1+cls_num+4)])

	for i in range(grid_num):
		for j in range(grid_num):
			grid_pos = [int(size[0]*i/grid_num), int(size[1]*j/grid_num)]
			grid_size = [
				int(size[0]*(i+1)/grid_num)-grid_pos[0],
				int(size[1]*(j+1)/grid_num)-grid_pos[1]]

			label_conf, label_cls, label_xywh = TransferLabelToConfClsXYWH(
				label[i][j])

			for b in range(bndbox_num):
				# if label_conf[b] < conf_th:
				# 	continue

				bndbox_pos = [
					int(grid_pos[0]+grid_size[0]*label_xywh[0][b][0]),
					int(grid_pos[1]+grid_size[1]*label_xywh[0][b][1])]
				bndbox_size = [
					int(size[0]*label_xywh[0][b][2]),
					int(size[1]*label_xywh[0][b][3])]

				point_a = (bndbox_pos[0]-bndbox_size[0]/2,
					 bndbox_pos[1]-bndbox_size[1]/2)
				point_b = (bndbox_pos[0]+bndbox_size[0]/2,
					 bndbox_pos[1]-bndbox_size[1]/2)
				point_c = (bndbox_pos[0]+bndbox_size[0]/2,
					 bndbox_pos[1]+bndbox_size[1]/2)
				point_d = (bndbox_pos[0]-bndbox_size[0]/2,
					 bndbox_pos[1]+bndbox_size[1]/2)

				img_draw.line([point_a, point_b, point_c, point_d, point_a],
					fill = (int(255 * label_conf[b]), 0, 0), width = 0)

	img.save(dst_path)

################################################################################
################################################################################
################################################################################

class YOLOConv(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding = 0, stride = 1):
		super().__init__()

		self.conv = torch.nn.Conv2d(
			in_channels = in_channels,
			out_channels = out_channels,
			kernel_size = kernel_size,
			padding = padding,
			stride = stride,
			bias = False).to(dev)

		self.bn = torch.nn.BatchNorm2d(
			num_features = out_channels,
			momentum = momentum).to(dev)

		self.leaky_relu = torch.nn.LeakyReLU(0.1).to(dev)

	def forward(self, x):
		return self.leaky_relu(self.bn(self.conv(x)))

################################################################################
################################################################################
################################################################################

class Inception(torch.nn.Module):
	def __init__(self, layers, sum_layer):
		super().__init__()
		self.layers = layers
		self.sum_layer = sum_layer

	def forward(self, x):
		return self.sum_layer([i(x) for i in self.layers])

################################################################################
################################################################################
################################################################################

class Concatenation(torch.nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		return torch.cat(x, self.dim)

################################################################################
################################################################################
################################################################################

class YOLOv1(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

		self.layers = torch.nn.Sequential(
			YOLOConv(3, 64, 7, padding = 3, stride = 2),
			torch.nn.MaxPool2d(kernel_size = 2, stride = 2),

			YOLOConv(64, 192, 3, padding = 1),
			torch.nn.MaxPool2d(kernel_size = 2, stride = 2),

			YOLOConv(192, 128, 1),
			YOLOConv(128, 256, 3, padding = 1),
			YOLOConv(256, 256, 1),
			YOLOConv(256, 512, 3, padding = 1),
			torch.nn.MaxPool2d(kernel_size = 2, stride = 2),

			Inception(torch.nn.ModuleList([
				torch.nn.Sequential(
					YOLOConv(512, 256, 1), YOLOConv(256, 512, 3, padding = 1)),
				torch.nn.Sequential(
					YOLOConv(512, 256, 1), YOLOConv(256, 512, 3, padding = 1)),
				torch.nn.Sequential(
					YOLOConv(512, 256, 1), YOLOConv(256, 512, 3, padding = 1)),
				torch.nn.Sequential(YOLOConv(512, 256, 1), YOLOConv(256, 512, 3, padding = 1))]),

				Concatenation(1)
			),

			YOLOConv(2048, 512, 1),
			YOLOConv(512, 1024, 3, padding = 1),
			torch.nn.MaxPool2d(kernel_size = 2, stride = 2),

			Inception(torch.nn.ModuleList([
				torch.nn.Sequential(
					YOLOConv(1024, 512, 1), YOLOConv(512, 1024, 3, padding = 1)),
				torch.nn.Sequential(
					YOLOConv(1024, 512, 1), YOLOConv(512, 1024, 3, padding = 1))]),

				Concatenation(1)
			),

			YOLOConv(2048, 1024, 3, padding = 1),
			YOLOConv(1024, 1024, 3, padding = 1, stride = 2),

			YOLOConv(1024, 1024, 3, padding = 1),
			YOLOConv(1024, 1024, 3, padding = 1),

			torch.nn.AdaptiveAvgPool2d(1),
			torch.nn.Flatten(),
			torch.nn.Linear(1024, 4096),
			torch.nn.Linear(4096, grid_num*grid_num *
							bndbox_num*(1+cls_num+4))
		)

		self.cls_act_func = torch.nn.Softmax(2)
		self.conf_xywh_act_func = torch.nn.Sigmoid()

	def forward(self, x):
		x_conf, x_cls, x_xywh = TransferLabelToConfClsXYWH_(self.layers(self.transform(x)))

		x_conf = self.conf_xywh_act_func(x_conf)
		x_cls = self.cls_act_func(x_cls)
		x_xywh = self.conf_xywh_act_func(x_xywh)

		return torch.reshape(
			torch.cat([x_conf, x_cls, x_xywh], 2),
			[-1, grid_num*grid_num*bndbox_num*(1+cls_num+4)])

################################################################################
################################################################################
################################################################################

class YOLOv1LossFunction(torch.nn.Module):
	def forward(self, pr, gt):
		loss = 0

		batch_size_ = gt.size(0)

		gt_conf, gt_cls, gt_xywh = TransferLabelToConfClsXYWH(gt)
		pr_conf, pr_cls, pr_xywh = TransferLabelToConfClsXYWH(pr)

		print(gt_conf.size())
		print(gt_cls.size())
		print(gt_conf[11])
		print(gt_cls[11])

		# [batch_size][grid_num*grid_num*bndbox_num*(1+cls_num+4)]
		# conf : [batch_size*grid_num*grid_num][bndbox]
		# cls  : [batch_size*grid_num*grid_num][bndbox][cls_num]
		# xywh : [batch_size*grid_num*grid_num][bndbox][4]

################################################################################

		for grid in range(gt_conf.size(0)):
			obj_num = torch.count_nonzero(gt_conf[grid])

			if obj_num == 0:
				loss += lambda_noobj*torch.sum(pr_conf[grid])
				continue

			for b in range(bndbox_num):
				loss += (pr_conf[grid][b]-1)**2

			for pr_i in range(bndbox_num):
				rsp_gt_i = -1
				rsp_gt_iou = 0

				for gt_i in range(obj_num):
					iou = IoU_two_xywh(
						pr_xywh[grid][pr_i], gt_xywh[grid][gt_i])

					if (rsp_gt_i == -1) | (rsp_gt_iou < iou):
						rsp_gt_i = gt_i
						rsp_gt_iou = iou

				print("rsp_gt_iou", rsp_gt_iou)

				loss += lambda_coord*(
					(pr_xywh[grid][pr_i][0]-gt_xywh[grid][rsp_gt_i][0])**2 +
					(pr_xywh[grid][pr_i][1]-gt_xywh[grid][rsp_gt_i][1])**2 +
					(math.sqrt(pr_xywh[grid][pr_i][2])-math.sqrt(gt_xywh[grid][rsp_gt_i][2]))**2 +
					(math.sqrt(pr_xywh[grid][pr_i][3])-math.sqrt(gt_xywh[grid][rsp_gt_i][3]))**2
				)

				"""
				a = torch.nn.functional.mse_loss(
					pr_cls[grid][pr_i], gt_cls[grid][rsp_gt_i])
				"""				

				loss += torch.nn.functional.mse_loss(
					pr_cls[grid][pr_i], gt_cls[grid][rsp_gt_i])

		return loss / batch_size_

################################################################################
################################################################################
################################################################################

class VOC2007(torch.utils.data.Dataset):
	def __init__(self, path, train = False):
		self.path = path
		self.train = train
		self.train_path = self.path+"/train"
		self.test_path = self.path+"/test"
		self.train_dataset = [i.strip(".jpg") for i in os.listdir(self.train_path+"/image")]
		self.test_dataset = [i.strip(".jpg") for i in os.listdir(self.test_path+"/image")]

	def __getitem__(self, index):
		path = self.train_path if self.train else self.test_path
		name = (self.train_dataset if self.train else self.test_dataset)[index]
		return LoadImage(path+"/image/"+name+".jpg"), LoadLabel(path+"/label/"+name+".xml")

	def __len__(self):
		return len(self.train_dataset) if self.train else len(self.test_dataset)

################################################################################
################################################################################
################################################################################

def GetLocalTime():
	return time.localtime()

def GetTimeLabel(localtime = None):
	if localtime == None:
		localtime = GetLocalTime()
	return time.strftime("%Y-%m-%d-%H%M", localtime)

def EpochStartLabel(epoch):
	return "{}    start    epoch {: >3}".format(time_label(), epoch)

def EpochEndLabel(epoch, loss):
	return "{}    end      epoch {: >3}    loss {:e}".format(time_label(), epoch, 8)

class Log:
	def __init__(self, path):
		self. path = path
		self.et = ET.ElementTree(file = path)
		self.root = self.et.getroot()
		self.train_log = self.root.find("train")
		self.test_log = self.root.find("test")
				
	def LogTrain(self, epoch, begin_time, end_time):
		log = ET.SubElement(self.train_log, "{:0>3}".format(epoch))
		log.tail = "\n"

		ET.SubElement(log, "time_begin").text = begin_time
		ET.SubElement(log, "time_end").text = end_time
		ET.SubElement(log, "learning_rate").text = "{:e}".format(learning_rate)
		ET.SubElement(log, "momentum").text = "{:e}".format(momentum)
		
		self.et.write(self.path)
	
	def LogTest(self, net, loss):
		log = ET.SubElement(self.test_log, "log")
		log.tail = "\n"

		ET.SubElement(log, "net").text = net
		ET.SubElement(log, "loss").text = "{:e}".format(loss)
		
		self.et.write(self.path)