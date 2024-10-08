export OUTPUT_DIR=./bert_hn_512_none_results/
# 模型名+提升模版类型+max_seq_len+相似度计算方式
python3 run_base_prompt.py \
    --output_dir=$OUTPUT_DIR \
    --prompt_type=hn \
    --select_arg_strategy=no_filter \
    # --matching_style=product_cosine \
    --matching_style=none \
    --cosine_space_dim=64 \
    --cosine_slices=128 \
    --cosine_factor=4 \
    --model_type=bert \
    --model_checkpoint=/root/HistoryRE/models/bert-base-chinese \
    --train_file=/root/HistoryRE/dataset/historyRE/train.txt \
    # --train_file_with_cos=../../data/train_filtered_with_cos.json \
    --dev_file=/root/HistoryRE/dataset/historyRE/val.txt \
    --test_file=/root/HistoryRE/dataset/historyRE/test.txt \
    # --train_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_train_related_info_0.75.json \
    # --dev_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_dev_related_info_0.75.json \
    # --test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json \
    # --pred_test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json \
    --sample_strategy=no \
    # --neg_top_k=3 \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=1 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42
# python3 run_base_prompt.py \
#     --output_dir=$OUTPUT_DIR \
#     --prompt_type=hn \
#     --select_arg_strategy=no_filter \
#     # --matching_style=product_cosine \
#     --matching_style=none \
#     --cosine_space_dim=64 \
#     --cosine_slices=128 \
#     --cosine_factor=4 \
#     --model_type=roberta \
#     --model_checkpoint=../../PT_MODELS/roberta-large/ \
#     --train_file=../../data/train_filtered.json \
#     --train_file_with_cos=../../data/train_filtered_with_cos.json \
#     --dev_file=../../data/dev_filtered.json \
#     --test_file=../../data/test_filtered.json \
#     --train_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_train_related_info_0.75.json \
#     --dev_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_dev_related_info_0.75.json \
#     --test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_gold_test_related_info_0.75.json \
#     --pred_test_simi_file=../../data/KnowledgeExtraction/simi_files/simi_omni_epoch_3_test_related_info_0.75.json \
#     --sample_strategy=corefnm \
#     --neg_top_k=3 \
#     --max_seq_length=512 \
#     --learning_rate=1e-5 \
#     --num_train_epochs=10 \
#     --batch_size=4 \
#     --do_train \
#     --warmup_proportion=0. \
#     --seed=42