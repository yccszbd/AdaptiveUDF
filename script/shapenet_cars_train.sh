#!/usr/bin/env bash
# train_all.sh —— 单文件完整日志版（含总运行时间）
# 用法：bash train_all.sh

# 0. 把文件里的 CRLF 转成 LF（保险起见）
sed -i 's/\r$//' "$0"

set -euo pipefail

# 记录总运行的起始时间（秒）
start_time=$(date +%s)


# 1. 参数列表
dataname=(
# 27d42437168ccd7ddd75f724c0ccbe00
# 2a9b4308929f91a6e1007bcfcf09901
# 3bfa196d1b48f1e0c5dccef20baf829a
# 4235a8f0f7b92ebdbfea8bc24170a935
# 666beb2570f33c64f64801ad2940cdd5
# b8f6994a33f4f1adbda733a39f84326d
# bc75e8adfee9ad70bda733a39f84326d
# f34c03711c3fc44ac10e9d4ee4bae4f4
f4440b2cecde4131afe1d4530f4c6e24
fffb1660a38af30ba4cf3601fb6b2442
)

# 2. 统一日志文件（log 文件夹）
mkdir -p log    # 如果 log 文件夹不存在就新建

logfile="log/$(date +"%Y-%m-%d_%H-%M-%S")_famous.log"

echo "===== $(date) : 开始全部训练，日志写入 ${logfile} =====" >> "$logfile"

for h in "${dataname[@]}"; do 
    echo "===== $(date) : 启动 shapenetCars ${h} =====" >> "$logfile"
    stdbuf -oL -eL python run.py \
        --gpu 1 \
        --conf "confs/shapenet_cars.conf" \
        --dir "shapenetCars" \
        --dataname "$h" \
        2>&1 | tee -a "$logfile"
    echo "===== $(date) : 结束 shapenetCars ${h} =====" >> "$logfile"
done

# 计算总耗时
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# 转换成 时:分:秒 格式
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "===== $(date) : 全部训练完成，日志保存在 ${logfile} =====" >> "$logfile"
echo "===== 总运行时间: ${hours}h ${minutes}m ${seconds}s (共 ${elapsed} 秒) =====" | tee -a "$logfile"

# === 训练结束后调用 Python 脚本，生成 Excel ===
python tools/log2csv.py "$logfile"
python tools/log2xlsx.py "$logfile"
