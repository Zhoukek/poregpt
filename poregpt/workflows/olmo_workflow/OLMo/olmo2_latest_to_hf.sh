source /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/set_env.sh  # 你之前那个脚本

export PYTHONPATH=/mnt/zzbnew/rnamodel/zhoukexuan/poregpt

python3 scripts/convert_olmo2_to_hf.py --input_dir "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/olmo_workflow/output_20m_ctx1280-lr5e4-vqe_distill003/steps/latest-unsharded" --output_dir "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/workflows/olmo_workflow/output_20m_ctx1280-lr5e4-vqe_distill003/hf_latest" --tokenizer_json_path "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/poregpt/dolma/tokenizer.json"
