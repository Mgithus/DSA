
!cd
!pwd






#creating a new directory for caring out :

import os

# Define the name of the directory you want to create
new_directory_name = "brats_23_swinunetr"

# Get the current working directory path
current_directory = os.getcwd()

# Create the full path for the new directory
new_directory_path = os.path.join(current_directory, new_directory_name)

# Check if the directory already exists
if not os.path.exists(new_directory_path):
    # Create the directory
    os.mkdir(new_directory_path)
    print(f"Directory '{new_directory_name}' created successfully.")
else:
    print(f"Directory '{new_directory_name}' already exists.")




#pip install -r requirements.txt



!git clone https://github.com/Mgithus/SWIN.git



#checking contents of directory:
contents = os.listdir(new_directory_path)
print("Contents of 'data' directory:", contents)








#FINETUNING


#for pretraining load model.pt from 
https://github.com/Mgithus/SWIN/blob/main/SwinUNETR/BRATS21/swin_unetr_brats21_segmentation_3d.ipynb


#Finetuning on single GPU with gradient check-pointing and without AMP

#To finetune a Swin UNETR model on a single GPU on fold 1 with gradient check-pointing and without amp, the model path using pretrained_dir and model name using --pretrained_model_name need to be provided:

!python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp --pretrained_model_name=<model-name> \
--pretrained_dir=<model-dir> --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48

#Finetuning on multi-GPU with gradient check-pointing and without AMP
#To finetune a Swin UNETR base model on multi-GPU on fold 1 with gradient check-pointing and without amp, the model path using pretrained_dir and model name using --pretrained_model_name need to be provided:

!python main.py --json_list=<json-path> --distributed --data_dir=<data-path> --val_every=5 --noamp --pretrained_model_name=<model-name> \
--pretrained_dir=<model-dir> --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48


#TRAINING

#Training from scratch on single GPU with gradient check-pointing and without AMP
#To train a Swin UNETR from scratch on a single GPU with gradient check-pointing and without AMP:

!python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48


#Training from scratch on multi-GPU with gradient check-pointing and without AMP
#To train a Swin UNETR from scratch on multi-GPU for 300 epochs with gradient check-pointing and without AMP:

!python main.py --json_list=<json-path> --data_dir=<data-path> --max_epochs=300 --val_every=5 --noamp --distributed \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48

#Training from scratch on multi-GPU without gradient check-pointing
#To train a Swin UNETR from scratch on multi-GPU without gradient check-pointing:

!python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --distributed \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --feature_size=48













import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch
print_config()




directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)
    
    
    
def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader

#########################################################################################################################

############################################################################################################






def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg



def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )



        
if __name__ == "__main__":
    

    data_dir = "//home//dlrs//Desktop//swin unetr//short_DATA_for_swin_unetr"
    json_list = "//home//dlrs//Desktop//swin unetr//short_DATA_for_swin_unetr//brats21_folds_shortdata.json"
    roi = (128, 128, 128)
    batch_size = 1
    sw_batch_size = 1
    fold = 1
    infer_overlap = 0.5
    max_epochs = 100
    val_every = 10
    train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)



    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    

    start_epoch = 0

    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )


    print(f"train completed, best average dice: {val_acc_max:.4f} ")

















#Plot the loss and Dice metric
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
plt.xlabel("epoch")
plt.plot(trains_epoch, loss_epochs, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_avg, color="green")
plt.show()
plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_tc, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_wt, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
plt.xlabel("epoch")
plt.plot(trains_epoch, dices_et, color="purple")
plt.show()



/home/dlrs/Desktop/short_2023data/BraTS-GLI-00703-000/BraTS-GLI-00703-000-t1n.nii.gz


#Create test set dataloader
case_num = "01619"

test_files = [
    {
        "image": [
            os.path.join(
                data_dir,
                "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-" + case_num + "/BraTS-GLI-" + case_num + "-t1n.nii.gz",
            ),
            os.path.join(
                data_dir,
                "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-" + case_num + "/BraTS-GLI-" + case_num + "-t1c.nii.gz",
            ),
            os.path.join(
                data_dir,
                "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-" + case_num + "/BraTS-GLI-" + case_num + "-t2f.nii.gz",
            ),
            os.path.join(
                data_dir,
                "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-" + case_num + "/BraTS-GLI-" + case_num + "-t2w.nii.gz",
            ),
        ],
        "label": os.path.join(
            data_dir,
            "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-" + case_num + "/BraTS-GLI-" + case_num + "-seg.nii.gz",
        ),
    }
]

test_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

test_ds = data.Dataset(data=test_files, transform=test_transform)

test_loader = data.DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)




#Load the best saved checkpoint and perform inference
We select a single case from the validation set and perform inference to compare the model segmentation output with the corresponding label.

model.load_state_dict(torch.load(os.path.join(root_dir, "model.pt"))["state_dict"])
model.to(device)
model.eval()

model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)


with torch.no_grad():
    for batch_data in test_loader:
        image = batch_data["image"].cuda()
        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4






#Visualize segmentation output and compare with label
slice_num = 67
img_add = os.path.join(
    data_dir,
    "TrainingData/BraTS2021_" + case_num + "/BraTS2021_" + case_num + "_t1ce.nii.gz",
)
label_add = os.path.join(
    data_dir,
    "TrainingData/BraTS2021_" + case_num + "/BraTS2021_" + case_num + "_seg.nii.gz",
)
img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()
plt.figure("image", (18, 6))
plt.subplot(1, 3, 1)
plt.title("image")
plt.imshow(img[:, :, slice_num], cmap="gray")
plt.subplot(1, 3, 2)
plt.title("label")
plt.imshow(label[:, :, slice_num])
plt.subplot(1, 3, 3)
plt.title("segmentation")
plt.imshow(seg_out[:, :, slice_num])
plt.show()
