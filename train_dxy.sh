python train.py \
    --model_cfg config/config_512_layer4.json \
    --output_dir saved_checkpoints/dxy/config_512_layer4_k_8 \
    --data_dir data/dxy \
    --vocab_file data/dxy/ontology.json \
    --result_output_path result.json \
    --add_task_1 \
    --add_task_2 \
    --add_task_3 \
    --add_task_4 \
    --add_middle_task_1 \
    --k 8 \
    --log_dir log

# epoch_max: 87
# sym_recall:0.912568306010929, disease:0.875, avg_turn:16.23076923076923

# sh train_dxy.sh