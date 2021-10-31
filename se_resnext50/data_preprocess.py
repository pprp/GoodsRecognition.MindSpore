import os
import cv2
import re 

train_dir_path = "/home/niu/data/all/train"
test_dir_path = "/home/niu/data/all/test"

def rename_folder(dir_path):
    # 规则: 将中文中可能不合法的命名处理
    # 1. 中文括号和英文括号替换为下划线
    # 2. 将空格替换为下划线
    # 3. 将.替换为下划线
    for idx, subdir in enumerate(os.listdir(dir_path)):
        new_folder = subdir.replace('（','_').replace('）','').replace(' ', '_').replace('.', '_').replace('(', '_').replace(')', '')
        whole_subdir = os.path.join(dir_path, subdir)
        whole_newdir = os.path.join(dir_path, new_folder)
        if len(subdir.split()) > 1:
            print(f"rename {whole_subdir} to {whole_newdir}")
        os.rename(whole_subdir, whole_newdir)


def rename_img(dir_path):
    # 将图片重命名为数字编号，防止中文出现
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

def grep_log(logs):
    res = re.findall("`.+'", logs)
    if res:
        print(f"res: {res}")
        return res 
    else:
        return None 

def rm(rm_list):
    if rm_list is not None:
        rm_list = [item.lstrip("`").rstrip("'") for item in rm_list]
        for item in rm_list:
            print("rm: ", item)
            os.system("rm %s" % item)

def main():
    # run with nohup
    # 1. rm nohup.out
    # 2. nohup python convert_png.py
    
    rename_folder(train_dir_path)
    rename_folder(test_dir_path)
    
    rename_img(train_dir_path)
    rename_img(test_dir_path)
    
    convert_png2jpg(train_dir_path)
    convert_png2jpg(test_dir_path)

if __name__ == "__main__":
    main()
