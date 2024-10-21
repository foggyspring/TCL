# /home/ubuntu/dataset/calvin/task_D_D/training/lang_annotations   地址 task id
import numpy as np
file_path = "/home/ubuntu/dataset/calvin/task_D_D/training/lang_annotations/auto_lang_ann.npy"
data = np.load(file_path, allow_pickle=True).item()
task_ids = data['language']['task']
# for task_id in task_ids:
#     print(task_id)
unique_ids = set(task_ids)
print(unique_ids)
num = len(unique_ids)
print(num)  #一共有34个task

# 创建任务名称到索引的映射
task_to_index = {task: idx for idx, task in enumerate(unique_ids)}

# 打印映射关系
print("Task to Index Mapping:", task_to_index)

# 将原始任务ID映射到整数编码
encoded_task_ids = [task_to_index[task] for task in task_ids]

# # 打印编码后的任务ID
# # print("Encoded task IDs:", encoded_task_ids)
    
    
# #我们的langencoder 需要的output_feature number 是task的数量 ，这里是34
# #相当于我们先训练一个lang_encoder，这个encoder可以输出预测当前task的id 
# #相同的id是相同的group

# #对于这个encoder我们需要使用cross entropy loss来训练
# # 定义交叉熵损失函数
# loss_fn = torch.nn.CrossEntropyLoss()

# # 计算损失
# loss = loss_fn(predicted_probs, target_labels)
# print("Cross Entropy Loss:", loss.item())

# if self.use_lang_vis_temporal_contrastive_loss:
#     self.task_id_pre = None
# #如何从当前的task中获得groundtruth？
#  gt_tasks = [
#             self.task_to_id[self.lang_data_val["language"]["task"][self.val_dataset.lang_lookup[i]]] for i in idx
#         ]
 
#  idx = Batch["idx"]

train_dataset = self.trainer.datamodule.train_datasets["lang"]  # type: ignore
            val_dataset = self.trainer.datamodule.val_datasets["lang"]  # type: ignore
            self.val_dataset = val_dataset
            lang_data_train = np.load(
                train_dataset.abs_datasets_dir / train_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
            ).item()
            self.lang_data_val = np.load(
                val_dataset.abs_datasets_dir / val_dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True
            ).item()
            lang_embeddings_val = np.load(
                val_dataset.abs_datasets_dir / val_dataset.lang_folder / "embeddings.npy", allow_pickle=True
            ).item()
            train_lang_instructions = list(set(lang_data_train["language"]["ann"]))
            train_lang_ids = [
                lang_data_train["language"]["ann"].index(instruction) for instruction in train_lang_instructions
            ]
            
            train_lang_tasks = list(np.array(lang_data_train["language"]["task"])[train_lang_ids])
            train_lang_task_ids = [list(set(train_lang_tasks)).index(task) for task in train_lang_tasks]

            self.task_to_id = {k: v for k, v in zip(set(train_lang_tasks), set(train_lang_task_ids))}
            self.train_lang_task_ids = np.array(train_lang_task_ids)
            



 

        
