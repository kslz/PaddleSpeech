#!/bin/bash
# -*- coding: utf-8-*-
import os
from subprocess import run

now_dir = os.path.abspath("./")
print(now_dir)
MAIN_ROOT = './../../..'
UTILS_PATH = MAIN_ROOT + "utils/"
LC_ALL = "C"
PYTHONDONTWRITEBYTECODE = 1
PYTHONIOENCODING = "UTF-8"
PYTHONPATH = MAIN_ROOT
MODEL = "fastspeech2"
BIN_DIR = MAIN_ROOT + f"/paddlespeech/t2s/exps/{MODEL}"

gpus = 0
stage = 0
stop_stage = 100

conf_path = "conf/default.yaml"
train_output_path = "exp/default"
ckpt_name = "snapshot_iter_153.pdz"

def main():
    # gen_duration_from_textgrid()
    preprocess()

    pass



def run_cmd(cmd_str="", echo_print=1):
    """ 执行cmd """
    if echo_print == 1:
        print("\n执行cmd指令：'{}'".format(cmd_str))
    run(cmd_str, shell=True)




def gen_duration_from_textgrid():
    # 从MFA的结果中获取持续时间
    # python3 ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
    #         --inputdir=./baker_alignment_tone \
    #         --output=durations.txt \
    #         --config=${config_path}

    print("从MFA的结果中获取持续时间")
    get_MFA_time = f"python {MAIN_ROOT}/utils/gen_duration_from_textgrid.py \
            --inputdir=./baker_alignment_tone \
            --output=durations.txt \
            --config={conf_path}"
    run_cmd(get_MFA_time)

def preprocess():
    # ==========================================================================
    # python3 ${BIN_DIR}/preprocess.py \
    #         --dataset=baker \
    #         --rootdir=~/datasets/BZNSYP/ \
    #         --dumpdir=dump \
    #         --dur-file=durations.txt \
    #         --config=${config_path} \
    #         --num-cpu=20 \
    #         --cut-sil=True
    print("提取特征")
    preprocess_str = f"python {BIN_DIR}/preprocess.py \
            --dataset=baker \
            --rootdir=datasets/BZNSYP/ \
            --dumpdir=dump \
            --dur-file=durations.txt \
            --config={conf_path} \
            --num-cpu=20 \
            --cut-sil=True"
    run_cmd(preprocess_str)
    print("特征已提取到dump目录下")


# 导入命令行选项

# 准备数据
# ==========================================================================
# ./local/preprocess.sh ${conf_path}





if __name__ == "__main__":
    main()
