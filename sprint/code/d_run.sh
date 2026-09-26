#!/bin/bash
# usage: d_run.sh <base|conf|ctrl|eown|eshuf|/abs/adapter/path> <kn|gen> [name]
m=$1; g=$2
MA="pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.80,max_model_len=4096"
case $m in
  conf) MA="$MA,enable_lora=True,max_lora_rank=16,lora_local_path=$HOME/reality-monitoring/adapters/r3_conf2_lr1e-4_rf0.5_s0";;
  ctrl) MA="$MA,enable_lora=True,max_lora_rank=16,lora_local_path=$HOME/reality-monitoring/adapters/r3_control2_lr1e-4_rf0.3_s0";;
  eown) MA="$MA,enable_lora=True,max_lora_rank=16,lora_local_path=$HOME/reality-monitoring/sprint/res_e/ad_own_s0";;
  eshuf) MA="$MA,enable_lora=True,max_lora_rank=16,lora_local_path=$HOME/reality-monitoring/sprint/res_e/ad_shuffled_s0";;
  /*) MA="$MA,enable_lora=True,max_lora_rank=16,lora_local_path=$m"; m=${3:-custom};;
esac
export HF_DATASETS_CACHE=$TMPDIR/ds
if [ "$g" = kn ]; then
  lm-eval run --model vllm --model_args "$MA" --tasks mmlu truthfulqa_mc2 --limit 100 --batch_size auto --output_path res_d/${m}_kn100 --apply_chat_template
else
  lm-eval run --model vllm --model_args "$MA" --tasks gsm8k ifeval --batch_size auto --output_path res_d/${m}_gen --fewshot_as_multiturn --apply_chat_template
fi
