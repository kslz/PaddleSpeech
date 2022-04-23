# 逐渐了解paddlespeech

## csmsc/tts3

### MFA-Montreal Forced Aligner

https://zhuanlan.zhihu.com/p/386884727



## 流程

### run.sh

设置路径等变量

#### path.sh

设置各种变量

### ${MAIN_ROOT}/utils/parse_options.sh

读参数用

### ./local/preprocess.sh

调用预处理脚本

#### ${MAIN_ROOT}/utils/gen_duration_from_textgrid.py

从 inputdir/speaker/* 目录下提取MFA信息 存放在durations.txt下

#### ${BIN_DIR}/preprocess.py

提取特征 存放在dump文件夹下