from YOLOv1_util import *
import torch
import os
import time

epoch_begin = 0
epoch_end = 70

net = torch.load("{:0>3d}.net".format(epoch_begin)).to(dev)
net.train()

loss_func = YOLOv1LossFunction()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

dataset = VOC2007("../VOC2007", train = True)
dataloader = Data.DataLoader(
	dataset = dataset, batch_size = batch_size, shuffle = True)

dataset_size = len(dataset)

log = Log("log.xml")

for epoch in range(epoch_begin+1, epoch_end+1):
	time_begin = GetTimeLabel()

	data_num_trained = 0

	for img, gt in dataloader:
		img_dev = img.to(dev)
		gt_dev = gt.to(dev)

		optimizer.zero_grad()

		pr = net(img_dev)

		loss = loss_func(pr, gt_dev)
		loss.backward()

		optimizer.step()

		data_num_trained += img_dev.size(0)

		print("epoch {}, {: >4} / {}, loss {:e}".format(
			epoch, data_num_trained, dataset_size, loss))
	
	time_end = GetTimeLabel()
	log.LogTrain(epoch, time_begin, time_end)

	print("time_begin {}".format(time_begin))
	print("time_end   {}".format(time_end))

	net_dst = "{:0>3d}.net".format(epoch)

	if epoch % 10 == 0:
		print("save as {}".format(net_dst))
		torch.save(net, net_dst)
	
	print("\n\n")