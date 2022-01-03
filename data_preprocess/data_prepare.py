#encoding=utf-8
import os
import cv2
import shutil
import re
import random
from numpy.lib.function_base import delete



def rename_folder(dir_path):
    for idx, subdir in enumerate(os.listdir(dir_path)):
        new_folder = (
            subdir.replace("（", "_")
            .replace("）", "")
            .replace(" ", "_")
            .replace(".", "_")
            .replace("(", "_")
            .replace(")", "")
        )
        whole_subdir = os.path.join(dir_path, subdir)
        whole_newdir = os.path.join(dir_path, new_folder)
        if len(subdir.split()) > 1:
            print(f"rename {whole_subdir} to {whole_newdir}")
        os.rename(whole_subdir, whole_newdir)


def rename_img(dir_path):
    for subdir in os.listdir(dir_path):
        whole_path = os.path.join(dir_path, subdir)
        for idx, img_name in enumerate(os.listdir(whole_path)):
            postfix = "png" if img_name.endswith("png") else "jpg"
            new_img_name = str(idx) + "." + postfix

            org_path = os.path.join(whole_path, img_name)
            new_path = os.path.join(whole_path, new_img_name)

            #     print(f"rename {org_path} to {new_path}")
            os.rename(org_path, new_path)


def convert_png2jpg(dir_path):
    # convert
    logs = ""

    for subdir in os.listdir(dir_path):
        whole_path = os.path.join(dir_path, subdir)
        for img_name in os.listdir(whole_path):
            img_whole_path = os.path.join(whole_path, img_name)
            if img_name.endswith("png"):
                # print(img_whole_path)
                img = cv2.imread(img_whole_path)
                cv2.imwrite(img_whole_path, img)
                os.system(
                    "convert %s %s"
                    % (img_whole_path, img_whole_path.replace("png", "jpg"))
                )
                # logs = logs + "".join(os.popen("convert %s %s"
                #      % (img_whole_path, img_whole_path.replace("png", "jpg"))))
                os.system("rm %s" % img_whole_path)

    # return logs

def delete_other(dir_path, num=7000):
    others_path = os.path.join(dir_path, "others")
    bak_path = others_path.replace("all", "all_bak")
    if not os.path.exists(bak_path):
        os.makedirs(bak_path)
    selected_files = random.choices(os.listdir(others_path), k=num)
    for sfile in selected_files:
        fsfile = os.path.join(others_path, sfile)
        # dsfile = os.path.join(bak_path, sfile)
        try:
            shutil.move(fsfile, bak_path)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        print(f"move {fsfile} to {bak_path}")

    print("DONE")


def sample_ten_percent_data(train_dir_path, sample_out_ratio=0.8):
    for item in os.listdir(train_dir_path):
        class_folder_path = os.path.join(train_dir_path, item)
        total_num_of_img = len(os.listdir(class_folder_path))

        target_folder_path = class_folder_path.replace("all", "all_sample_out")
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        # 采样80%的数据移出去。
        sampled_list = random.sample(os.listdir(class_folder_path), int(total_num_of_img * sample_out_ratio))
        
        for sampled_item in sampled_list:
            full_sampled_item_path = os.path.join(class_folder_path, sampled_item)
            target_path = full_sampled_item_path.replace("all", "all_sample_out")
            print(f"from {full_sampled_item_path} to {target_path}")
            # training move 
            shutil.move(full_sampled_item_path, target_path)


def get_num_class(dir_path):
    num_list = []
    for item in os.listdir(dir_path):
        fpath = os.path.join(dir_path, item)
        print(f"class:{item}'s number is {len(os.listdir(fpath))}")
        num_list.append(len(os.listdir(fpath)))

    print(sorted(num_list))


def normal_process(config):
    train_dir_path = config.data_path 
    test_dir_path = config.eval_data_path 

    # 重命名中文文件夹，替换括号，空格等内容
    rename_folder(train_dir_path)
    rename_folder(test_dir_path)

    # 重命名文件夹中的图片名称
    rename_img(train_dir_path)
    rename_img(test_dir_path)

    # 将png转化为jpg，并删除png
    convert_png2jpg(train_dir_path)
    convert_png2jpg(test_dir_path)


def main():
    train_dir_path = "/HOME/scz0088/run/all/train"
    test_dir_path = "/HOME/scz0088/run/all/test"

    # 重命名中文文件夹，替换括号，空格等内容
    rename_folder(train_dir_path)
    rename_folder(test_dir_path)

    # 重命名文件夹中的图片名称
    rename_img(train_dir_path)
    rename_img(test_dir_path)

    # 将png转化为jpg，并删除png
    convert_png2jpg(train_dir_path)
    convert_png2jpg(test_dir_path)



if __name__ == "__main__":
    main()
