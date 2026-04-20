OLMO_OUTPUT_PATH=output_300m_ctx1280-gbsz4096-lr1e4-vqe342s036000l1
python3 scripts/convert_olmo2_to_hf.py --input_dir "../$OLMO_OUTPUT_PATH/steps/latest-unsharded" --output_dir "../$OLMO_OUTPUT_PATH/hf_latest" --tokenizer_json_path "olmo_data/tokenizers/pore_16k/tokenizer.json"
