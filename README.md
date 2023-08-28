句子级事件抽取

1.数据预处理

bash run_duee_1.sh dara_prepare

2.训练

bash run_duee_1.sh trigger_train

bash run_duee_1.sh role_train

3.预测

bash run_duee_1.sh trigger_predict

bash run_duee_1.sh role_predict

6.数据后处理

将role处理结果和trigger处理结果合并

bash run_duee_1.sh pred_2_submit