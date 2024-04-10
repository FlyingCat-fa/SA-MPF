python train.py \
    --model_cfg config/config_512_layer4.json \
    --output_dir saved_checkpoints/mz4/config_512_layer4 \
    --data_dir data/mz4 \
    --vocab_file data/mz4/ontology.json \
    --result_output_path result.json \
    --add_task_1 \
    --add_task_2 \
    --add_task_3 \
    --add_task_4 \
    --add_middle_task_1 \
    --k 8 \
    --num_train_epochs 200 \
    --log_dir log

# epoch_max: 94
# sym_recall:0.8502202643171806, disease:0.7746478873239436, avg_turn:19.330985915492956


# sh train_mz4.sh