import os
import numpy as np
label_dir = r"F:\train40label"
lable_Files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
whole_max = -10000000000
whole_min = 10000000000
for num in range(len(lable_Files)):
    lable_path = os.path.join(label_dir,lable_Files[num])
    raw_file = np.fromfile(lable_path, dtype=np.float32)
    if np.max(raw_file)>=whole_max:
        whole_max = np.max(raw_file)
    if np.min(raw_file)<=whole_min:
        whole_min = np.min(raw_file)

print("max is %d"%whole_max)
print("min is %d"%whole_min)

label_dir_new = r"E:\WJC_train\sparse_proj_10slice_new"
for num in range(len(lable_Files)):
    lable_path = os.path.join(label_dir, lable_Files[num])
    lable_path2 = os.path.join(label_dir_new,lable_Files[num])
    raw_file = np.fromfile(lable_path, dtype=np.float32)
    raw_file_new = (raw_file-whole_min)/(whole_max-whole_min)
    raw_file_new.tofile(lable_path2)

