import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os, random, numpy as np
from test_DRAEM import test
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("precise")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    object_list = ['capsule', 'bottle', 'pill', 'transistor', 'cable', 'toothbrush', 'metal_nut', 'hazelnut', 'screw']

    for obj_name in obj_names:
        print(obj_name)

        # 如果是物体类别
        if obj_name in object_list:
            is_object = True
        else:
            is_object = False

        max_auc = 0
        max_auc_pixel = 0

        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, k=args.k, center_dim=args.center_dim, is_object=is_object)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=7, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[0.7*args.epochs,0.8*args.epochs],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        n_iter = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):

                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                normal_mask = torch.zeros_like(anomaly_mask).cuda()

                # 正常分支
                gray_rec, loss = model(gray_batch, is_normal=True, epoch=epoch)
                print("正常重构 nan?",torch.isnan(gray_rec).any())
                residual_nor = torch.sum(torch.abs(gray_batch - gray_rec), dim=1, keepdim=True)
                print("正常残差 nan?",torch.isnan(residual_nor).any())
                gray_joined = torch.cat((gray_rec, gray_batch, residual_nor), dim=1)
                print("正常拼接 nan?",torch.isnan(gray_joined).any())

                segment_loss_gray = torch.zeros(1).cuda()

                if epoch > 120:
                    gray_out_b1, gray_out_b2, gray_out_b3, gray_out_b4, gray_out_b5, gray_out_b6, gray_out_final = model_seg(gray_joined)
                    gray_out_b1_sm = torch.softmax(gray_out_b1, dim=1)
                    gray_out_b2_sm = torch.softmax(gray_out_b2, dim=1)
                    gray_out_b3_sm = torch.softmax(gray_out_b3, dim=1)
                    gray_out_b4_sm = torch.softmax(gray_out_b4, dim=1)
                    gray_out_b5_sm = torch.softmax(gray_out_b5, dim=1)
                    gray_out_b6_sm = torch.softmax(gray_out_b6, dim=1)
                    gray_out_final_sm = torch.softmax(gray_out_final, dim=1)

                    segment_loss_gray_b1 = loss_focal(gray_out_b1_sm, normal_mask) # 正常样本分割损失b1
                    segment_loss_gray_b2 = loss_focal(gray_out_b2_sm, normal_mask) # 正常样本分割损失b2
                    segment_loss_gray_b3 = loss_focal(gray_out_b3_sm, normal_mask) # 正常样本分割损失b3
                    segment_loss_gray_b4 = loss_focal(gray_out_b4_sm, normal_mask) # 正常样本分割损失b4
                    segment_loss_gray_b5 = loss_focal(gray_out_b5_sm, normal_mask) # 正常样本分割损失b5
                    segment_loss_gray_b6 = loss_focal(gray_out_b6_sm, normal_mask) # 正常样本分割损失b6
                    segment_loss_gray_final = loss_focal(gray_out_final_sm, normal_mask) # 正常样本分割损失final

                    segment_loss_gray = segment_loss_gray_b1 + segment_loss_gray_b2 + segment_loss_gray_b3 + segment_loss_gray_b4 + segment_loss_gray_b5 + segment_loss_gray_b6 + segment_loss_gray_final

                l2_loss_gray = loss_l2(gray_rec, gray_batch)  # 正常样本L2损失
                ssim_loss_gray = loss_ssim(gray_rec, gray_batch)  # 正常样本ssim损失

                Lc = loss['Lc'] # 中心约束损失
                kl_loss = loss['kl_loss'] # KL散度聚类损失
                entropy_loss = loss['entropy_loss'] # 熵损失
                limit_entropy_loss = loss['limit_entropy_loss'] # 避免大多数实例分配给同一个集群

                l2_loss_aug = torch.zeros(1).cuda()
                ssim_loss_aug = torch.zeros(1).cuda()
                l2_loss_aug_m = torch.zeros(1).cuda()
                ssim_loss_aug_m = torch.zeros(1).cuda()
                segment_loss_aug = torch.zeros(1).cuda()
                Ld = torch.zeros(1).cuda()

                if epoch > 120:
                    # 异常分支
                    aug_rec, _ = model(aug_gray_batch, is_normal=False, epoch=epoch)
                    print("异常重构 nan?", torch.isnan(aug_rec).any())
                    residual_ano = torch.sum(torch.abs(aug_gray_batch - aug_rec), dim=1, keepdim=True)
                    print("异常残差 nan?", torch.isnan(residual_ano).any())
                    aug_joined = torch.cat((aug_rec, aug_gray_batch, residual_ano), dim=1)

                    aug_out_b1, aug_out_b2, aug_out_b3, aug_out_b4, aug_out_b5, aug_out_b6, aug_out_final = model_seg(aug_joined)
                    aug_out_b1_sm = torch.softmax(aug_out_b1, dim=1)
                    aug_out_b2_sm = torch.softmax(aug_out_b2, dim=1)
                    aug_out_b3_sm = torch.softmax(aug_out_b3, dim=1)
                    aug_out_b4_sm = torch.softmax(aug_out_b4, dim=1)
                    aug_out_b5_sm = torch.softmax(aug_out_b5, dim=1)
                    aug_out_b6_sm = torch.softmax(aug_out_b6, dim=1)
                    aug_out_final_sm = torch.softmax(aug_out_final, dim=1)

                    l2_loss_aug = loss_l2(aug_rec, gray_batch) # 异常样本L2损失
                    ssim_loss_aug = loss_ssim(aug_rec, gray_batch) # 异常样本ssim损失

                    # coef = torch.sum(torch.ones_like(anomaly_mask)) / torch.sum(anomaly_mask)
                    # l2_loss_aug_m = coef * loss_l2(anomaly_mask * aug_rec, anomaly_mask * gray_batch)  # 聚焦于异常区域的L2损失
                    # ssim_loss_aug_m = coef * loss_ssim(anomaly_mask * aug_rec, anomaly_mask * gray_batch)  # 聚焦于异常区域的ssim损失

                    segment_loss_aug_b1 = loss_focal(aug_out_b1_sm, anomaly_mask)  # 异常样本分割损失b1
                    segment_loss_aug_b2 = loss_focal(aug_out_b2_sm, anomaly_mask)  # 异常样本分割损失b2
                    segment_loss_aug_b3 = loss_focal(aug_out_b3_sm, anomaly_mask)  # 异常样本分割损失b3
                    segment_loss_aug_b4 = loss_focal(aug_out_b4_sm, anomaly_mask)  # 异常样本分割损失b4
                    segment_loss_aug_b5 = loss_focal(aug_out_b5_sm, anomaly_mask)  # 异常样本分割损失b5
                    segment_loss_aug_b6 = loss_focal(aug_out_b6_sm, anomaly_mask)  # 异常样本分割损失b6
                    segment_loss_aug_final = loss_focal(aug_out_final_sm, anomaly_mask)  # 异常样本分割损失final

                    segment_loss_aug = segment_loss_aug_b1 + segment_loss_aug_b2 + segment_loss_aug_b3 + segment_loss_aug_b4 + segment_loss_aug_b5 + segment_loss_aug_b6 + segment_loss_aug_final


                # 合并同类损失
                l2_loss = l2_loss_gray + l2_loss_aug
                ssim_loss = ssim_loss_gray + ssim_loss_aug
                segment_loss = segment_loss_gray + segment_loss_aug

                print("l2_loss:",l2_loss,"ssim_loss:",ssim_loss,"segment_loss:",segment_loss,"Lc:",Lc,"kl_loss:",kl_loss,"entropy_loss:",entropy_loss,"limit_entropy_loss:",limit_entropy_loss)
                # total_loss = l2_loss + ssim_loss + segment_loss + Lc + kl_loss - 0.1* entropy_loss + Ld
                # total_loss = 1000 * l2_loss + 1000 * ssim_loss + 1000 * segment_loss + 0.1 * Lc + 0.1 * kl_loss - 0.1 * entropy_loss + 0.1 * Ld
                total_loss = 1000 * l2_loss + 1000 * ssim_loss + 1000 * segment_loss + 0.1 * Lc + 0.1 * kl_loss + 0.1 * entropy_loss - 0.1 * limit_entropy_loss

                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 20 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                    visualizer.plot_loss(Lc, n_iter, loss_name='Lc_loss')
                    visualizer.plot_loss(kl_loss, n_iter, loss_name='kl_loss')
                    visualizer.plot_loss(entropy_loss, n_iter, loss_name='entropy_loss')
                    visualizer.plot_loss(limit_entropy_loss, n_iter, loss_name='limit_entropy_loss')
                    visualizer.plot_loss(total_loss, n_iter, loss_name='total_loss')
                if args.visualize and n_iter % 20 == 0:
                    # 解锁正常分割+异常分支
                    if epoch > 120:
                        aug_b1_mask = aug_out_b1_sm[:, 1:, :, :]
                        aug_b2_mask = aug_out_b2_sm[:, 1:, :, :]
                        aug_b3_mask = aug_out_b3_sm[:, 1:, :, :]
                        aug_b4_mask = aug_out_b4_sm[:, 1:, :, :]
                        aug_b5_mask = aug_out_b5_sm[:, 1:, :, :]
                        aug_b6_mask = aug_out_b6_sm[:, 1:, :, :]
                        aug_final_mask = aug_out_final_sm[:, 1:, :, :]
                        visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='augmented')
                        visualizer.visualize_image_batch(gray_batch, n_iter, image_name='augmented_recon_target')
                        visualizer.visualize_image_batch(aug_rec, n_iter, image_name='augmented_recon_out')
                        visualizer.visualize_image_batch(residual_ano, n_iter, image_name='augmented_residual')
                        visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='augmented_mask_target')
                        visualizer.visualize_image_batch(aug_final_mask, n_iter, image_name='augmented_mask_final')
                        visualizer.visualize_image_batch(aug_b6_mask, n_iter, image_name='augmented_mask_b6')
                        visualizer.visualize_image_batch(aug_b5_mask, n_iter, image_name='augmented_mask_b5')
                        visualizer.visualize_image_batch(aug_b4_mask, n_iter, image_name='augmented_mask_b4')
                        visualizer.visualize_image_batch(aug_b3_mask, n_iter, image_name='augmented_mask_b3')
                        visualizer.visualize_image_batch(aug_b2_mask, n_iter, image_name='augmented_mask_b2')
                        visualizer.visualize_image_batch(aug_b1_mask, n_iter, image_name='augmented_mask_b1')


                        gray_b1_mask = gray_out_b1_sm[:, 1:, :, :]
                        gray_b2_mask = gray_out_b2_sm[:, 1:, :, :]
                        gray_b3_mask = gray_out_b3_sm[:, 1:, :, :]
                        gray_b4_mask = gray_out_b4_sm[:, 1:, :, :]
                        gray_b5_mask = gray_out_b5_sm[:, 1:, :, :]
                        gray_b6_mask = gray_out_b6_sm[:, 1:, :, :]
                        gray_final_mask = gray_out_final_sm[:, 1:, :, :]

                        visualizer.visualize_image_batch(normal_mask, n_iter, image_name='normal_mask_target')
                        visualizer.visualize_image_batch(gray_final_mask, n_iter, image_name='normal_mask_final')
                        visualizer.visualize_image_batch(gray_b6_mask, n_iter, image_name='normal_mask_b6')
                        visualizer.visualize_image_batch(gray_b5_mask, n_iter, image_name='normal_mask_b5')
                        visualizer.visualize_image_batch(gray_b4_mask, n_iter, image_name='normal_mask_b4')
                        visualizer.visualize_image_batch(gray_b3_mask, n_iter, image_name='normal_mask_b3')
                        visualizer.visualize_image_batch(gray_b2_mask, n_iter, image_name='normal_mask_b2')
                        visualizer.visualize_image_batch(gray_b1_mask, n_iter, image_name='normal_mask_b1')

                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='normal')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='normal_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='normal_recon_out')
                    visualizer.visualize_image_batch(residual_nor, n_iter, image_name='normal_residual')


                n_iter +=1

            scheduler.step()

            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))

            if epoch > 200 and epoch % 5 == 0:
                print("epoch:", epoch)
                pa = argparse.ArgumentParser()
                pa.add_argument('--gpu_id', action='store', type=int, default=1)
                pa.add_argument('--base_model_name', action='store', type=str, default="DRAEM_test_0.0001_800_bs4")
                pa.add_argument('--data_path', action='store', type=str, default="../mvtec_anomaly_detection/")
                pa.add_argument('--checkpoint_path', action='store', type=str, default="./checkpoints/")

                pa.add_argument('--k', action='store', default=10)  # 聚类中心数量
                pa.add_argument('--center_dim', action='store', default=256)  # 聚类中心的维度

                ar = pa.parse_args()

                obj_list = [obj_name]

                with torch.cuda.device(ar.gpu_id):
                    auroc, auroc_pixel, ap, ap_pixel = test(obj_list, ar.data_path, ar.checkpoint_path,
                                                            ar.base_model_name, ar)

                    if auroc + auroc_pixel > max_auc + max_auc_pixel:
                        max_auc = auroc
                        max_auc_pixel = auroc_pixel
                        torch.save(model.state_dict(), os.path.join("./best_checkpoints/", run_name + ".pckl"))
                        torch.save(model_seg.state_dict(),
                                   os.path.join("./best_checkpoints/", run_name + "_seg.pckl"))
                        print("最大更新")

                    print(obj_name + " max_auc", max_auc)
                    print(obj_name + " max_auc_pixel", max_auc_pixel)

                    writer.add_scalar(tag=obj_name + " auroc", scalar_value=auroc, global_step=epoch)
                    writer.add_scalar(tag=obj_name + " auroc_pixel", scalar_value=auroc_pixel, global_step=epoch)
        writer.close()


if __name__=="__main__":
    seed = 42
    print("seed:", seed)
    setup_seed(seed)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=-1)
    parser.add_argument('--bs', action='store', type=int, default=4)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default=800)
    parser.add_argument('--gpu_id', action='store', type=int, default=1)
    parser.add_argument('--data_path', action='store', type=str, default="../mvtec_anomaly_detection/")
    parser.add_argument('--anomaly_source_path', action='store', type=str, default="../dtd/images/")
    parser.add_argument('--checkpoint_path', action='store', type=str, default="./checkpoints/")
    parser.add_argument('--log_path', action='store', type=str, default="./logs/")
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--k', action='store', default=10) # 聚类中心数量
    parser.add_argument('--center_dim', action='store', default=256) # 聚类中心的维度

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = [#'capsule',
                     # 'bottle',
                     'carpet',
                     # 'leather',
                     # 'pill',
                     'transistor',
                     # 'tile',
                     # 'cable',
                     # 'zipper',
                     'toothbrush',
                     # 'metal_nut',
                     # 'hazelnut',
                     # 'screw',
                     # 'grid',
                     # 'wood'
                    ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

