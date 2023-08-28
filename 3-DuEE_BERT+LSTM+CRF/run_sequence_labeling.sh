data_dir=$1
conf_path=$2
ckpt_dir=$3
predict_data=$4
learning_rate=$5
is_train=$6
max_seq_len=$7
batch_size=$8
epoch=${9}
pred_save_path=${10}
hidden_dim=${11} 
rnn_layers=${12} 
lstm_dropout_ratio=${13}
dropout=${14}
if [ "$is_train" = True ]; then
    python sequence_labeling.py \
                            --num_epoch ${epoch} \
                            --learning_rate ${learning_rate} \
                            --tag_path ${conf_path} \
                            --train_data ${data_dir}/train.tsv \
                            --dev_data ${data_dir}/dev.tsv \
                            --test_data ${data_dir}/test.tsv \
                            --predict_data ${predict_data} \
                            --do_train True \
                            --do_predict False \
                            --max_seq_len ${max_seq_len} \
                            --batch_size ${batch_size} \
                            --skip_step 10 \
                            --valid_step 50 \
                            --checkpoints ${ckpt_dir} \
                            --init_ckpt ${ckpt_dir}/best.pdparams \
                            --predict_save_path ${pred_save_path} \
                            --device gpu\
                            --hidden_dim ${hidden_dim}\
                            --rnn_layers ${rnn_layers}\
                            --lstm_dropout_ratio ${lstm_dropout_ratio}\
                            --dropout ${dropout}

else
    export CUDA_VISIBLE_DEVICES=0
    python sequence_labeling.py \
            --num_epoch ${epoch} \
            --learning_rate ${learning_rate} \
            --tag_path ${conf_path} \
            --train_data ${data_dir}/train.tsv \
            --dev_data ${data_dir}/dev.tsv \
            --test_data ${data_dir}/test.tsv \
            --predict_data ${predict_data} \
            --do_train False \
            --do_predict True \
            --max_seq_len ${max_seq_len} \
            --batch_size ${batch_size} \
            --skip_step 10 \
            --valid_step 50 \
            --checkpoints ${ckpt_dir} \
            --init_ckpt ${ckpt_dir}/best.pdparams \
            --predict_save_path ${pred_save_path} \
            --device gpu\
            --hidden_dim ${hidden_dim}\
            --rnn_layers ${rnn_layers}\
            --lstm_dropout_ratio ${lstm_dropout_ratio}\
            --dropout ${dropout}
fi
