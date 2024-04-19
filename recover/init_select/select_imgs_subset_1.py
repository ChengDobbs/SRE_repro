import pickle
import os
import shutil
from pathlib import Path

now_path = Path(__file__).parent
file_path = (now_path / "selected_samples.pkl").resolve()
# with open("/home/st2/select/init_imgnet/selected_samples.pkl", "rb") as f:
#     data_list = pickle.load(f)

# with open("init_select/selected_samples.pkl", "rb") as f:
#     data_list = pickle.load(f)

with open(file_path, "rb") as f:
    data_list = pickle.load(f)

imagenet_dir = "/raid/data_dujw/imagenet"
imagenet_dir_train = os.path.join(imagenet_dir, "train")
select_num = 50

for cls in range(len(data_list)):

    item = data_list[cls]
    for j in range(select_num):
        img_path = item[j][0]
        img_path_full = os.path.join(imagenet_dir_train, img_path.split('_')[0], img_path)
       
        if os.path.exists(img_path_full):
            # print("found:", img_path_full)
            
            class_label = img_path.split("_")[0]
            target_dir = os.path.join(imagenet_dir, "select/select_dataset_"+str(select_num), class_label)
            os.makedirs(target_dir, exist_ok=True)

            target_img_path = os.path.join(target_dir, os.path.basename(img_path))

            shutil.copy(img_path_full, target_img_path)

        else:
            print("not found:", img_path_full)
        
    print("class "+str(cls)+' done')

    