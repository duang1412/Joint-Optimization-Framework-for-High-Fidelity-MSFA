import os
from spectral import *
from scipy.io import savemat

# 设置路径
input_dir = 'F:/ICVL'
output_dir = 'F:/ICVL/selected_400_700nm'
os.makedirs(output_dir, exist_ok=True)

# 遍历目录中的所有 .hdr 文件
for file in os.listdir(input_dir):
    if file.endswith('.hdr'):
        hdr_path = os.path.join(input_dir, file)
        mat_name = os.path.splitext(file)[0] + '.mat'
        mat_path = os.path.join(output_dir, mat_name)

        try:
            # 打开 .hdr 文件对应的高光谱图像
            img = open_image(hdr_path)
            data = img.load()

            # 截取波段（ICVL 通常为8:253为400-700nm, 共245个波段）
            spectrum = data[:, :, 8:253]

            # 保存为 .mat 文件
            savemat(mat_path, {'rad': spectrum})
            print(f"Saved: {mat_path}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
