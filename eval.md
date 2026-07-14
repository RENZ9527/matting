# Full Fine-Tuning Matting Inference and Evaluation

本项目用于加载 **FLUX / Edit2Perceive 全参数微调模型**，并在多个图像抠图 benchmark 上执行推理和评估。

与 LoRA 推理不同，本脚本直接加载完整模型权重，不包含以下操作：

- 不执行 LoRA 权重格式转换
- 不创建 PEFT Adapter
- 不调用 `FluxLoRALoader`
- 不调用 `mapping_lora_state_dict`
- 不需要指定 LoRA Rank
- 直接将完整微调权重加载到 `pipe.dit`

默认支持以下三个 benchmark：

- P3M-500-NP
- AM-2K
- AIM-500

评估指标包括：

- MSE
- MAD
- SAD
- Gradient Error
- Connectivity Error

---

## 1. 文件说明

将推理与评估脚本保存为：

```text
inference_eval_full_finetune.py
```

推荐的项目结构如下：

```text
matting/
├── inference_eval_full_finetune.py
├── pipelines/
│   └── flux_image_new.py
├── models/
│   ├── unified_dataset.py
│   └── utils.py
├── utils/
│   ├── metric.py
│   └── eval_matting.py
└── data_split/
    ├── P3M_matting/
    │   └── filenames_val_NP.txt
    ├── AM_matting/
    │   └── filenames_val.txt
    └── AIM_matting/
        └── filenames_val.txt
```

运行脚本前，请进入项目根目录：

```bash
cd /nvmedata/workspace2/users/rzc/matting
```

这样脚本才能正确找到：

```text
./data_split/P3M_matting/filenames_val_NP.txt
./data_split/AM_matting/filenames_val.txt
./data_split/AIM_matting/filenames_val.txt
```

---

## 2. 环境要求

建议使用项目训练时相同的 Python 环境。

进入对应 Conda 环境：

```bash
conda activate your_environment
```

确认 PyTorch 和 CUDA 可以正常使用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

正常情况下应看到：

```text
True
NVIDIA ...
```

安装脚本可能需要的依赖：

```bash
pip install numpy pillow tqdm safetensors
```

主要依赖包括：

```text
torch
numpy
Pillow
tqdm
safetensors
```

如果权重是 `.pt`、`.pth` 或 `.bin`，则不一定需要 `safetensors`。

---

## 3. 数据集路径配置

脚本内部默认定义了三个 benchmark：

```python
DEFAULT_BENCHMARKS = {
    "p3m-np": BenchmarkConfig(
        name="p3m-np",
        gt_root="/nvmedata/workspace2/users/rzc/datasets/P3M-10k",
        file_list="./data_split/P3M_matting/filenames_val_NP.txt",
        output_subdir="p3m_np",
    ),

    "am": BenchmarkConfig(
        name="am",
        gt_root="/nvmedata/workspace2/users/rzc/datasets/AM-2k",
        file_list="./data_split/AM_matting/filenames_val.txt",
        output_subdir="am_2k",
    ),

    "aim": BenchmarkConfig(
        name="aim",
        gt_root="/nvmedata/workspace2/users/rzc/datasets/AIM-500",
        file_list="./data_split/AIM_matting/filenames_val.txt",
        output_subdir="aim_500",
    ),
}
```

请根据服务器上的实际位置修改：

```python
gt_root
file_list
```

例如：

```python
"am": BenchmarkConfig(
    name="am",
    gt_root="/your/path/to/AM-2k",
    file_list="/your/path/to/filenames_val.txt",
    output_subdir="am_2k",
)
```

---

## 4. 文件列表格式

每个 benchmark 的文件列表建议使用三列格式：

```text
merged_image_path trimap_path alpha_path
```

例如：

```text
merged/0001.jpg trimap/0001.png alpha/0001.png
merged/0002.jpg trimap/0002.png alpha/0002.png
```

脚本读取规则为：

```python
merged_path = items[0]
trimap_path = items[1]
alpha_path = items[-1]
```

因此，多于三列时：

- 第一列作为原图路径
- 第二列作为 trimap 路径
- 最后一列作为 alpha matte 路径

所有路径均相对于对应 benchmark 的 `gt_root`。

例如：

```text
GT_ROOT=/nvmedata/workspace2/users/rzc/datasets/AM-2k
```

文件列表中包含：

```text
merged/0001.jpg trimap/0001.png alpha/0001.png
```

实际读取路径为：

```text
/nvmedata/workspace2/users/rzc/datasets/AM-2k/merged/0001.jpg
/nvmedata/workspace2/users/rzc/datasets/AM-2k/trimap/0001.png
/nvmedata/workspace2/users/rzc/datasets/AM-2k/alpha/0001.png
```

---

## 5. 基础模型路径

默认基础模型路径为：

```text
/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev
```

执行时通过以下参数指定：

```bash
--model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev"
```

基础模型目录需要与训练时使用的模型保持一致。

---

## 6. 全参数微调权重格式

脚本支持传入单个权重文件，也支持传入 checkpoint 目录。

### 6.1 单个 Safetensors 文件

```text
/path/to/checkpoint-1900/model.safetensors
```

执行时：

```bash
--checkpoint "/path/to/checkpoint-1900/model.safetensors"
```

### 6.2 PyTorch 权重文件

支持：

```text
model.pt
model.pth
pytorch_model.bin
checkpoint.pt
checkpoint.pth
```

例如：

```bash
--checkpoint "/path/to/checkpoint-1900/model.pt"
```

### 6.3 Checkpoint 目录

也可以直接传入目录：

```bash
--checkpoint "/path/to/checkpoint-1900"
```

脚本会自动查找目录中的常见权重文件，例如：

```text
model.safetensors
diffusion_pytorch_model.safetensors
pytorch_model.bin
model.pt
model.pth
checkpoint.pt
```

也支持 Hugging Face 风格的分片权重，例如：

```text
model.safetensors.index.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

---

## 7. 首次测试

第一次执行时，建议每个 benchmark 只测试 10 个样本。

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/test_full_finetune_eval" \
    --benchmarks p3m-np am aim \
    --resolution 768 \
    --num_inference_steps 1 \
    --cfg_scale 1 \
    --seed 42 \
    --device cuda:0 \
    --dtype bf16 \
    --workers 4 \
    --max_samples 10
```

该命令会对三个 benchmark 分别运行前 10 个样本。

测试时重点观察权重加载日志：

```text
[weights] parameter_coverage=...
```

对于全参数微调权重，覆盖率通常应接近：

```text
100%
```

如果覆盖率明显低于 90%，脚本默认会终止，避免误用错误权重。

---

## 8. 完整运行三个 Benchmark

确认少量样本运行正常后，可以执行完整评测：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900" \
    --benchmarks p3m-np am aim \
    --resolution 768 \
    --num_inference_steps 1 \
    --cfg_scale 1 \
    --seed 42 \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

请将：

```text
/path/to/checkpoint-1900
```

替换为实际的全参数微调 checkpoint。

例如：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/nvmedata/workspace2/users/rzc/output/full_finetune/checkpoints/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900" \
    --benchmarks p3m-np am aim \
    --resolution 768 \
    --num_inference_steps 1 \
    --cfg_scale 1 \
    --seed 42 \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

---

## 9. 只运行指定 Benchmark

### 9.1 只运行 P3M-500-NP

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/output" \
    --benchmarks p3m-np \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

### 9.2 只运行 AM-2K

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/output" \
    --benchmarks am \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

### 9.3 只运行 AIM-500

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/output" \
    --benchmarks aim \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

### 9.4 同时运行两个 Benchmark

```bash
--benchmarks p3m-np am
```

---

## 10. 输出目录结构

假设设置：

```bash
--output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900"
```

执行完成后，目录结构类似：

```text
final_eval_full_finetune_1900/
├── p3m_np/
│   ├── merged/
│   │   ├── image_001.npy
│   │   └── image_002.npy
│   ├── inference_failures.json
│   └── evaluation_failures.txt
├── am_2k/
│   ├── merged/
│   │   ├── image_001.npy
│   │   └── image_002.npy
│   ├── inference_failures.json
│   └── evaluation_failures.txt
├── aim_500/
│   ├── merged/
│   │   ├── image_001.npy
│   │   └── image_002.npy
│   ├── inference_failures.json
│   └── evaluation_failures.txt
├── benchmark_results.json
└── benchmark_results.txt
```

预测结果会保留输入图像的相对目录结构，并将后缀替换为 `.npy`。

例如输入相对路径：

```text
merged/person/0001.jpg
```

预测输出为：

```text
p3m_np/merged/person/0001.npy
```

---

## 11. 结果文件说明

### 11.1 benchmark_results.txt

该文件提供方便阅读的汇总结果：

```text
Benchmark           Valid          MSE          MAD          SAD         Grad         Conn
p3m-np            500/500       0.0123       0.0234      10.1234       5.4321       4.3210
am               2000/2000       0.0111       0.0212       9.8765       5.1234       4.0123
aim                500/500       0.0109       0.0208       9.6543       4.9876       3.9876
```

### 11.2 benchmark_results.json

该文件包含更完整的信息，例如：

- checkpoint 路径
- 基础模型路径
- 推理参数
- 权重覆盖率
- 每个 benchmark 的推理数量
- 跳过数量
- 失败数量
- 最终指标

适合后续自动化汇总和画图。

### 11.3 inference_failures.json

记录推理失败的样本和异常信息。

### 11.4 evaluation_failures.txt

记录评估失败的样本，例如：

- 预测文件不存在
- alpha 文件不存在
- trimap 文件不存在
- 图像格式错误
- 指标计算出现异常

---

## 12. 断点续推

默认情况下，如果某个 `.npy` 预测文件已经存在且可以正常读取，脚本会跳过该样本。

因此，推理中断后可以直接重新执行相同命令：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/existing_output" \
    --benchmarks p3m-np am aim \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

已经完成的预测不会重新计算。

如果检测到 `.npy` 文件损坏，脚本会自动重新推理该样本。

---

## 13. 强制覆盖已有结果

需要重新生成全部预测时，加入：

```bash
--overwrite
```

完整示例：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/output" \
    --benchmarks p3m-np am aim \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32 \
    --overwrite
```

注意：该选项会重新生成所有已存在的预测结果。

---

## 14. 只评估已有预测

如果推理已经完成，只想重新计算指标，可使用：

```bash
--skip_inference
```

示例：

```bash
python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900" \
    --benchmarks p3m-np am aim \
    --skip_inference \
    --workers 32
```

在该模式下不会加载模型，也不会占用 GPU。

`--checkpoint` 参数仍然需要填写，但不会真正读取该权重。

如果只进行评估，可以填入原来的 checkpoint 路径。

---

## 15. 只推理，不评估

加入：

```bash
--skip_evaluation
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/path/to/output" \
    --benchmarks p3m-np am aim \
    --device cuda:0 \
    --dtype bf16 \
    --skip_evaluation
```

---

## 16. 数据类型选择

默认使用：

```bash
--dtype bf16
```

支持：

```text
bf16
fp16
fp32
```

### BF16

```bash
--dtype bf16
```

推荐在支持 BF16 的 GPU 上使用，例如较新的 A100、H100 或其他数据中心 GPU。

### FP16

```bash
--dtype fp16
```

如果 GPU 不支持 BF16，可尝试 FP16。

### FP32

```bash
--dtype fp32
```

显存占用较高，一般不建议用于大型 FLUX 模型推理。

---

## 17. 评估进程数量

评估阶段默认使用多个 CPU 进程：

```bash
--workers 32
```

根据机器 CPU 数量调整。

例如：

```bash
--workers 16
```

如果多进程出现共享内存不足、进程被杀死或系统负载过高，可以降低为：

```bash
--workers 8
```

调试时可以使用单进程：

```bash
--workers 1
```

---

## 18. 推理参数说明

### resolution

输入和输出分辨率：

```bash
--resolution 768
```

应与训练设置保持一致。

### num_inference_steps

扩散推理步数：

```bash
--num_inference_steps 1
```

当前任务原始代码使用 1 步推理。

### cfg_scale

Classifier-Free Guidance Scale：

```bash
--cfg_scale 1
```

建议与训练和原始推理配置保持一致。

### seed

随机种子：

```bash
--seed 42
```

### prompt

默认提示词：

```text
Transform to matting map while maintaining original composition
```

可以通过命令行覆盖：

```bash
--prompt "Transform to matting map while maintaining original composition"
```

### deterministic_flow

默认设置为：

```python
deterministic_flow=False
```

需要启用时加入：

```bash
--deterministic_flow
```

---

## 19. 权重覆盖率检查

脚本会自动比较 checkpoint 参数与 `pipe.dit` 参数。

运行时会输出类似：

```text
[weights] matched_tensors=...
[weights] parameter_coverage=99.98%
[weights] missing=...
[weights] unexpected=...
```

默认最低覆盖率：

```bash
--min_weight_coverage 0.90
```

即 90%。

如果覆盖率低于 90%，脚本会停止运行。这通常说明：

1. 加载了错误的 checkpoint。
2. checkpoint 不是完整的全参数权重。
3. checkpoint 只包含 LoRA 权重。
4. 基础模型配置与训练时不一致。
5. 参数名称前缀不兼容。
6. 实际微调的模块不是 `pipe.dit`。
7. checkpoint 保存的是整个 pipeline，而当前脚本只加载 `dit`。

仅用于调试时，可以适当降低：

```bash
--min_weight_coverage 0.50
```

但正式评估不建议降低。

---

## 20. 批量评估多个 Checkpoint

创建文件：

```text
eval_all_full_checkpoints.sh
```

内容如下：

```bash
#!/bin/bash
set -euo pipefail

BASE_DIR="/nvmedata/workspace2/users/rzc/output/full_finetune/checkpoints"
MODEL_ROOT="/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev"
EVAL_ROOT="/nvmedata/workspace2/users/rzc/eval_full_finetune"

mkdir -p "$EVAL_ROOT"

for CKPT_DIR in $(find "$BASE_DIR" \
    -maxdepth 1 \
    -type d \
    -name "checkpoint-*" | sort -V); do

    CKPT_NAME=$(basename "$CKPT_DIR")
    CKPT_STEP="${CKPT_NAME#checkpoint-}"
    OUT_ROOT="$EVAL_ROOT/checkpoint_${CKPT_STEP}"

    echo "============================================================"
    echo "Evaluating: $CKPT_NAME"
    echo "Checkpoint: $CKPT_DIR"
    echo "Output:     $OUT_ROOT"
    echo "============================================================"

    if [ -f "$OUT_ROOT/.eval_done" ]; then
        echo "$CKPT_NAME 已经完成，跳过。"
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
        --model_root "$MODEL_ROOT" \
        --checkpoint "$CKPT_DIR" \
        --output_root "$OUT_ROOT" \
        --benchmarks p3m-np am aim \
        --resolution 768 \
        --num_inference_steps 1 \
        --cfg_scale 1 \
        --seed 42 \
        --device cuda:0 \
        --dtype bf16 \
        --workers 32

    touch "$OUT_ROOT/.eval_done"

    echo "$CKPT_NAME 推理和评估完成。"
done

echo "所有 checkpoint 已处理完成。"
```

赋予执行权限：

```bash
chmod +x eval_all_full_checkpoints.sh
```

运行：

```bash
./eval_all_full_checkpoints.sh
```

也可以后台执行并保存日志：

```bash
nohup ./eval_all_full_checkpoints.sh \
    > eval_all_full_checkpoints.log 2>&1 &
```

查看日志：

```bash
tail -f eval_all_full_checkpoints.log
```

---

## 21. 使用指定 GPU

### 使用 GPU 0

```bash
CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py ...
```

同时参数设置：

```bash
--device cuda:0
```

### 使用物理 GPU 1

```bash
CUDA_VISIBLE_DEVICES=1 python inference_eval_full_finetune.py \
    ... \
    --device cuda:0
```

注意：设置 `CUDA_VISIBLE_DEVICES=1` 后，物理 GPU 1 在当前进程中会映射为逻辑设备 `cuda:0`。

因此推荐始终使用：

```bash
--device cuda:0
```

不要写成：

```bash
CUDA_VISIBLE_DEVICES=1 ... --device cuda:1
```

---

## 22. 后台执行单个 Checkpoint

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/path/to/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900" \
    --benchmarks p3m-np am aim \
    --resolution 768 \
    --num_inference_steps 1 \
    --cfg_scale 1 \
    --seed 42 \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32 \
    > full_finetune_1900_eval.log 2>&1 &
```

查看进程：

```bash
ps -ef | grep inference_eval_full_finetune.py
```

查看日志：

```bash
tail -f full_finetune_1900_eval.log
```

查看 GPU：

```bash
nvidia-smi
```

---

## 23. 常见问题

### 23.1 找不到文件列表

报错示例：

```text
FileNotFoundError: 找不到文件列表
```

原因通常是运行目录错误。

请先进入项目目录：

```bash
cd /nvmedata/workspace2/users/rzc/Edit2Perceive
```

再执行脚本。

或者将 `file_list` 修改为绝对路径。

---

### 23.2 找不到数据集图片

报错可能记录在：

```text
inference_failures.json
```

检查：

```bash
cat /path/to/output/p3m_np/inference_failures.json
```

确认文件列表中的路径是否相对于 `gt_root`。

---

### 23.3 权重覆盖率过低

示例：

```text
全参数权重覆盖率只有 10%，低于要求的 90%
```

检查以下内容：

- checkpoint 是否为全参数微调权重
- 是否误传了 LoRA 权重
- 基础模型是否与训练时一致
- 权重实际微调对象是否为 `pipe.dit`
- checkpoint 中参数是否带特殊前缀
- checkpoint 是否只保存了部分模块

可以打印 checkpoint 参数名进行检查：

```python
from safetensors.torch import load_file

state_dict = load_file("/path/to/model.safetensors")

for key in list(state_dict.keys())[:100]:
    print(key)
```

---

### 23.4 CUDA Out of Memory

可尝试：

1. 确认使用 BF16：

   ```bash
   --dtype bf16
   ```

2. 关闭其他 GPU 进程。

3. 降低分辨率进行调试：

   ```bash
   --resolution 512
   ```

4. 使用显存更大的 GPU。

5. 只运行一个 benchmark：

   ```bash
   --benchmarks p3m-np
   ```

评估阶段的 `--workers` 只影响 CPU，不会明显降低模型 GPU 显存。

---

### 23.5 Multiprocessing 报错

尝试将：

```bash
--workers 32
```

改为：

```bash
--workers 8
```

或者：

```bash
--workers 1
```

脚本已经使用：

```python
if __name__ == "__main__":
    mp.freeze_support()
    main()
```

不要删除主程序入口保护。

---

### 23.6 所有指标都是 NaN

检查：

- 预测 `.npy` 是否存在
- 预测 shape 是否正确
- alpha 路径是否正确
- trimap 路径是否正确
- `compute_matting_metrics` 是否接受当前 trimap 数值范围
- 文件列表最后一列是否确实是 alpha 路径

查看：

```text
evaluation_failures.txt
```

---

### 23.7 预测结果范围异常

脚本会自动处理：

- `[0, 1]`
- `[0, 255]`
- RGB 输出
- 单通道输出
- BCHW
- BHWC
- CHW
- HWC

最终预测会被转换为：

```text
float32
shape = [H, W]
range = [0, 1]
```

---

## 24. 推荐的正式执行命令

```bash
cd /nvmedata/workspace2/users/rzc/Edit2Perceive

CUDA_VISIBLE_DEVICES=0 python inference_eval_full_finetune.py \
    --model_root "/nvmedata/workspace2/share_model/FLUX.1-Kontext-dev" \
    --checkpoint "/nvmedata/workspace2/users/rzc/output/full_finetune/checkpoints/checkpoint-1900" \
    --output_root "/nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900" \
    --benchmarks p3m-np am aim \
    --prompt "Transform to matting map while maintaining original composition" \
    --resolution 768 \
    --num_inference_steps 1 \
    --cfg_scale 1 \
    --seed 42 \
    --device cuda:0 \
    --dtype bf16 \
    --workers 32
```

运行完成后查看：

```bash
cat /nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900/benchmark_results.txt
```

或者：

```bash
cat /nvmedata/workspace2/users/rzc/final_eval_full_finetune_1900/benchmark_results.json
```

---

## 25. 查看命令行帮助

```bash
python inference_eval_full_finetune.py --help
```

该命令会列出所有支持的参数及其说明。