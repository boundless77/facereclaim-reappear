import os
import shutil
from tqdm import tqdm
  # #
# src_folder = '/path/to/your/de_faceswap_20k_224/256/target/'
# dst_folder = '/path/to/your/attribute/'
#
  # #
# if not os.path.exists(dst_folder):
#     os.makedirs(dst_folder)
#
  # # 00001_source.jpg20000_source.jpg
# for i in tqdm(range(20001, 40001)):
#     src_file = os.path.join(src_folder, f'{i:05d}_target.jpg')
#     dst_file = os.path.join(dst_folder, f'{i:05d}_target.jpg')
#
  # #
#     if os.path.exists(src_file):
#         shutil.copy(src_file, dst_file)

print("File copy operation completed.")

folder = '/path/to/your/attribute/'


for filename in os.listdir(folder):
  # "_source""_target"
    new_filename = filename.replace('_source', '').replace('_target', '')


    src_file = os.path.join(folder, filename)
    dst_file = os.path.join(folder, new_filename)


    os.rename(src_file, dst_file)

print("File rename operation completed.")
