import time
import torch.nn as nn
import torch,argparse,os
import net,config,loaddataset,myDataSet


# train stage one
def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = "cuda"
        # DEVICE ="cpu"
        # DEVICE = torch.device("cuda:" + str(config.gpu_name))
        # 每次训练计算图改动较小使用，在开始前选取较优的基础算法（比如选择一种当前高效的卷积算法）
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset=myDataSet.MyData("/data2/wyz/SimCLR/Val_Data/",28800,True)#201555)#412720)
    train_data=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=12 , drop_last=True)

    model =net.SimCLRStage1(256)
    model = nn.DataParallel(model)
    model.to(DEVICE)
    # model.load_state_dict(torch.load("./pth/model_stage1_epoch15.pth", map_location='cpu'), strict=False)
    lossLR=net.Loss().to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-6)

    os.makedirs(config.save_path, exist_ok=True)
    print("开始训练！")
    aa = time.time()  # 开始时间
    for epoch in range(1,args.max_epoch+1):
        model.train()
        total_loss = 0
        print(len(train_data))
        for batch,(genuNUM,imgL,imgR,labels) in enumerate(train_data):
            # print(imgL.cpu().shape)
            if(batch%1==0):
                bb = time.time()  # 结束时间.
                cc = bb - aa
                print("第" + str(epoch) + "个epoch" + ":用了" + str(cc // 1) + "秒！")

            imgL,imgR,labels=imgL.to(DEVICE),imgR.to(DEVICE),labels.to(DEVICE)
            print(imgL.cpu().shape)
            _, pre_L=model(imgL)
            _, pre_R=model(imgR)

            loss=lossLR(pre_L,pre_R,args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
            total_loss += loss.detach().item()

        bb = time.time()  # 结束时间.
        cc = bb - aa
        print("第" + str(epoch) + "个epoch" + ":用了" + str(cc // 1) + "秒！")

        print("epoch loss:",total_loss/len(train_dataset)*args.batch_size)

        with open(os.path.join(config.save_path, "stage1_loss.txt"), "a") as f:
            f.write(str(total_loss/len(train_dataset)*args.batch_size) + " ")
        # print(os.path.join("./pth/", 'model_stage1_epoch' + str(epoch) + '.pth'))
        torch.save(model.state_dict(), os.path.join("/data2/wyz/SimCLR/pth/", 'model_stage1_epoch' + str(epoch) + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--max_epoch', default=1000, type=int, help='')

    args = parser.parse_args()
    train(args)
