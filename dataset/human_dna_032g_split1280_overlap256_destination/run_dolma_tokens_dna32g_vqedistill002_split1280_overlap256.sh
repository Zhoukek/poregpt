dolma tokens \
    --documents "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256/validation/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_64k/tokenizer.json" \
    --destination /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256_destination/validation \
    --dtype "uint32" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
    
dolma tokens \
    --documents "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256/test/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_64k/tokenizer.json" \
    --destination /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256_destination/test \
    --dtype "uint32" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
 
dolma tokens \
    --documents "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256/train/*.gz" \
    --tokenizer.name_or_path "/mnt/si003067jezr/default/poregpt/dolma/tokenizers/pore_64k/tokenizer.json" \
    --destination /mnt/zzbnew/rnamodel/zhoukexuan/poregpt/dataset/human_dna_032g_split1280_overlap256_destination/train \
    --dtype "uint32" \
    --tokenizer.pad_token_id 1 \
    --tokenizer.bos_token_id 2 \
    --tokenizer.eos_token_id 3 \
    --processes 32
 
