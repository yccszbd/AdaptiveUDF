#!/usr/bin/env bash
# train_all.sh  ——  单文件完整日志版
# 用法：bash train_all.sh

# 0. 把文件里的 CRLF 转成 LF（保险起见）
sed -i 's/\r$//' "$0"

set -euo pipefail

# 1. 参数列表
steps=( 1 3 5  )
dataname=(
    anchor
    daratech
    dc
    gargoyle
    lord_quas
)

# 2. 统一日志文件（log 文件夹）
mkdir -p log    # 如果 log 文件夹不存在就新建

logfile="log/$(date +"%Y-%m-%d_%H-%M-%S")_srb.log"

echo "===== $(date) : 开始全部训练，日志写入 ${logfile} =====" >> "$logfile"

# # 3. 过滤 tqdm 进度条的正则
# filter_re='^\s*[0-9]+%\|.*\[.*\].*\|.*[0-9]+/[0-9]+\s*\[.*\]$'

for h in "${dataname[@]}"; do 
    for g in "${steps[@]}"; do
        echo "===== $(date) : 启动 srb${g}  ${h} =====" >> "$logfile"
        stdbuf -oL -eL python run.py \
            --gpu 0 \
            --conf "confs/srb${g}.conf" \
            --dir "srb" \
            --dataname "$h" \
            2>&1 | tee -a "$logfile"
        echo "===== $(date) : 结束 srb${g}  ${h} =====" >> "$logfile"
    done
done

echo "===== $(date) : 全部训练完成，日志保存在 ${logfile} =====" >> "$logfile"

# === 训练结束后调用 Python 脚本，生成 Excel ===
python tools/log2csv.py "$logfile"