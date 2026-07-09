import os
import random
from collections import defaultdict
import shutil

# 数据目录
data_dir = 'F:/ICVL/selected_400_700nm'
all_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

# 分组：scene -> [file1, file2, ...]
scene_dict = defaultdict(list)
for fname in all_files:
    if '-' in fname:
        scene = fname.split('-')[0]  # e.g., 'rsh_0406'
        scene_dict[scene].append(fname)

# 固定随机种子确保可复现
random.seed(42)

train_set, test_set = [], []

# 每个场景分配部分样本到测试集中
for scene, files in scene_dict.items():
    files = sorted(files)
    random.shuffle(files)
    n = len(files)
    n_test = max(1, int(n * 0.2))  # 每个场景至少一个测试样本
    test_set.extend(files[:n_test])
    train_set.extend(files[n_test:])

# 精确控制总数量
train_set = train_set[:161]
test_set = test_set[:40]

print(f"Train: {len(train_set)} images")
print(f"Test:  {len(test_set)} images")

# 创建 test 文件夹
test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# 将测试集文件移入 test 目录
for fname in test_set:
    src = os.path.join(data_dir, fname)
    dst = os.path.join(test_dir, fname)
    if os.path.exists(src):
        shutil.move(src, dst)
