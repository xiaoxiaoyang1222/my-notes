### 提交格式

Please download the submission example and do not change the readme.txt. You can add the score into the prediction.csv file and do not change its name



### 数据集

 你这个 real_data 的文件作用如下：

  - train_condition-label.csv
    训练标注主表。每行一条样本，字段有：video_path/type/model_name/condition_path + 4 个质量分 + AnomalyTimestamp。
    当前统计：863 条训练样本。
  - train
    训练视频本体目录（按 type/model_name/*.mp4 组织）。
    通过 train_condition-label.csv 里的 video_path 去定位。
  - test
    待提交测试视频目录。
    当前模板里有 298 条测试样本。
  - condition
    条件输入库（你问的这个目录核心用途就是给 ConditionVideoAlignment 用）。
    由每行的 condition_path 指向，按模态对应：
    text-2D -> .txt（文本条件）
    image-4D -> .jpg（图像条件）
    video-4D -> .mp4（视频条件）
    训练和测试都会引用这个目录里的文件。
  - submission (1).zip
    官方提交模板，里面有 prediction.csv 和 readme.txt。
    实际提交时应保留行顺序和 video_path/type/model_name/condition_path，并覆盖预测列。
  - readme.txt
    你本地历史输出的说明文件，不是训练标签文件。
  - __MACOSX
    解压残留目录，可忽略。

  补充两个你数据里的小边界点：

  - condition_path 基本都存在且可解析，但有 2 条是空字符串（video-4D），代码里要做缺失兜底。
  - condition 文件数量比样本数多是正常的：它是“全量条件池”，训练和测试共同引用，且存在少量重复/共享条件。

### 训练

data/model/InternVideo-main/InternVideo2/multi_modality/demo/train_quality_head_lovif_first_strict.py 训练代码

### 思考的优化方向

训练和推理时间都较快，是否适当提高模型大小会有更好的效果？

### 最好成绩记录

data/sunny/LoViF_record/submitt/20260314_084344_internvideo2_v2_strict_train_only/20260314_084344_internvideo2_v2_strict_train_only.zip

### GPT5.4

明天思路：把训练代码，竞赛连接，指标都给他，数据集内容目录都给他，问问这个还有什么修改建议，然后用5.3codex去执行一下，看结果

（但也别忽视之前说的训练问题，这也是一种提升思路：比如condition 折叠时间等问题）



（还有就是等到时候测试集下来了，记得把验证集的标签给加上去验证验证，别从训练集里分了，然后文件名改一改，路径名都改一改）

### 评估指标

参与者需提交能够在以下四个维度准确评估生成视频的指标模型和奖励模型：

- **Video Quality Score:** Evaluates the overall visual quality of the generated videos, including clarity, color, noise, blur, and other visual artifacts. High scores indicate clear images, natural colors, and no obvious defects.
  **视频质量评分：** 评估生成视频的整体视觉质量，包括清晰度、色彩、噪点、模糊及其他视觉伪影。高分表示图像清晰、色彩自然且无明显缺陷。
- **Physical Realism Score:** Evaluates whether the motion, collisions, gravity, fluid dynamics, and other physical phenomena in the generated videos conform to physical laws. High scores indicate realistic and natural physical effects without violations of physical principles. Participants must also identify timestamps where physical violations occur.
  **物理真实感评分：** 评估生成视频中的运动、碰撞、重力、流体力学及其他物理现象是否符合物理定律。高分表示物理效果真实且自然，且不违反物理原则。参与者还必须识别发生身体违规的时间戳。
- **Condition-Video Alignment Score:** Evaluates how well the generated videos match their input conditions (text prompts, input images, or input videos). High scores indicate that the video content fully conforms to the descriptions or requirements specified in the input conditions.
  **条件-视频对齐评分：** 评估生成视频与输入条件（文本提示、输入图片或输入视频）的匹配程度。高分表示视频内容完全符合输入条件中规定的描述或要求。
- **Spatial-Temporal Consistency Score:** Evaluates the coherence of videos in both temporal and spatial dimensions, including whether object shapes are stable, motion is continuous, and scenes are consistent. High scores indicate no flickering, frame skipping, object deformation, or other consistency issues.
  **时空一致性评分：** 评估视频在时间和空间维度上的连贯性，包括物体形状是否稳定、运动是否连续以及场景是否一致。高分表示没有闪烁、跳帧、物体变形或其他一致性问题。

In addition to the four scoring dimensions above, participants must also provide:
除了上述四个评分维度外，参与者还必须提供：

- **Physical Anomaly Detection IOU:** Measures the accuracy of identifying specific timestamps where physical violations occur in generated videos. This includes detecting violations of dynamics, optics, thermodynamics, and other physical principles. Timestamps should be provided in the format such as "00:02-00:05"， or None if no violations are present..
  **物理异常检测欠条：** 衡量识别生成视频中发生物理违规的特定时间戳的准确性。这包括检测动力学、光学、热力学及其他物理原理的违背。时间戳应以“00：02-00：05”格式提供，若无违规则为“无”。

The final ranking will be determined by a composite score that balances all evaluation dimensions. The weighted formula for the initial results is:
最终排名将通过综合评分决定，该分数平衡了所有评估维度。初始结果的加权公式为：
**Final_Score = 0.2 × IOU + 0.4 × SRCC (4 dimensions, average) + 0.4 × PLCC (4 dimensions, average)**.
**Final_Score = 0.2 × IOU + 0.4 × SRCC（4 维，平均）+ 0.4 × PLCC（4 维，平均）。**



### 可行的数据增强方法总结

#### 1）多视角时间增强 train_quality_head_lovif_first_strict_temporal_aug.py，（失败）

不是只取一组 clip，而是每个视频训练时生成两个视角：

- `view_global`：全局 sparse clip
- `view_local`：局部 dense clip

把两个 view 的特征做：

- concat
- attention fusion
- 或均值融合

#### 对时间戳头

优先用 `view_local`

这样会明显优于你现在“单个全局 4 帧 pooled feature”。



#### 2）（训练时间相对增强）伪样本扩增：同一个训练视频做多次不同时间采样，生成多份特征当作伪扩增样本来训练。data/model/InternVideo-main/InternVideo2/multi_modality/demo/train_quality_head_lovif_first_strict_pseudo_aug.py（失败！！！！！！）

这个特别适合你当前流程。

因为你现在是先抽特征，再训练 head。
 那你完全可以对**同一个训练视频**，生成多份特征缓存：

例如每个视频存 4 份：

- `feat_aug1.npy`
- `feat_aug2.npy`
- `feat_aug3.npy`
- `feat_aug4.npy`

它们标签相同，但时间采样不同。

#### 好处

- 不用改太多训练框架
- 直接把样本数扩大 4 倍
- 对时间鲁棒性提升很明显

这比做强图像增强靠谱得多。

------

#### 3）时间戳监督增强

你现在时间戳任务最好不要只用一个 start/end 回归。

可以把时间戳标签扩成：

- `has_anomaly`
- `segment_mask`
- `soft boundary`

### 具体做法

假设视频被分成 `T=8` 段。
 把 GT 区间 `[s,e]` 投影到 8 段上，得到一个长度 8 的二值向量：

```
[0,0,1,1,1,0,0,0]
```

然后再做一个 soft 版本：

```
[0.0,0.1,0.8,1.0,0.7,0.1,0.0,0.0]
```

这本质上也是一种标签增强。
 它会比直接回归 start/end 稳很多。







### flash_attn

cuda：12.5





## . 学习率 `--lr` 与 防过拟合 `--patience`

- **注意点**：你的训练集只有 **863** 个视频，这在深度学习里属于“极小数据集”。
- 在当前的设置下（`--lr 3e-4`，`--patience 35`），从你刚才发我的日志来看，模型一般在第 40 到 90 个 Epoch 之间就会触发早停（Early Stopping），这说明模型很容易在训练集上“背下答案”并开始在验证集上掉分。
- **建议**：如果你加入了 CLIP 特征后发现模型过拟合非常快（比如第 20 个 Epoch 就早停了），你可以尝试在命令行把学习率调小一点：`--lr 1e-4`，或者把 `--weight-decay` 加大到 `1e-3` 来增强正则化。

## 2. 隐藏层维度 `--hidden-dim` (模型脑容量)

- **注意点**：在纯视频特征时，输入是 `526` 维，你设置 `--hidden-dim 1024` 是完全足够的（甚至是翻倍的冗余，网络很好学）。
- **升级多模态后**：一旦加上了 CLIP 特征（512维），你的输入总维度会暴涨到 `1038` 维。如果此时发现训练时 Loss 下降得很慢，或者在验证集上分数上不去（欠拟合），**建议在终端启动训练时加上 `--hidden-dim 2048`**。这会让 `QualityHeadV2` 的神经元数量翻倍，有足够的能力去消化多模态的交叉信息。









python /mnt/data/gemini_infer_improved_v3.py \
  --head-weights /data/sunny/LoViF_record/model_train/你的run目录/weights.pt \
  --submission-template /data/sunny/submission_real.zip \
  --test-root /data/sunny/real_data/test \
  --model-path /data/model/InternVideo2-Stage2_6B





python /mnt/data/model/InternVideo2-Stage2_6B/code/gpt_infer.py \
  --head-weights /data/sunny/LoViF_record/model_train/data/sunny/LoViF_record/model_train/20260318_075913_internvideo2_6b_multimodal_v4_duration_thr/weights.pt \
  --submission-template /data/sunny/submission_real.zip \
  --test-root /data/sunny/real_data/test \
  --model-path /data/model/InternVideo2-Stage2_6B \
  --force-refresh-video-cache \
  --force-refresh-condition-cache \
  --force-refresh-duration-cache













刚才实际运行参数如下。

  训练命令：

  python /data/model/InternVideo2-Stage2_6B/code/physcore_train_infer.py train \
    --data_root /data/sunny/real_data \
    --output_dir /data/model/InternVideo2-Stage2_6B/code/physcore_runs_baseline \
    --folds 5 \
    --epochs 25 \
    --batch_size 64 \
    --num_workers 8 \
    --video_frames 16 \
    --cond_video_frames 8

  训练里使用的默认参数（未显式传入）：

  - --lr 1e-3
  - --weight_decay 1e-4
  - --seed 42
  - --force_recache False

  推理命令：

  python /data/model/InternVideo2-Stage2_6B/code/physcore_train_infer.py infer \
    --data_root /data/sunny/real_data \
    --model_dir /data/model/InternVideo2-Stage2_6B/code/physcore_runs_baseline \
    --submission_zip '/data/sunny/real_data/submission (1).zip' \
    --video_frames 16 \
    --cond_video_frames 8









python /data/model/InternVideo2-Stage2_6B/code/physcore_train_infer.py train \
  --data_root /data/sunny/real_data \
  --output_dir /data/model/InternVideo2-Stage2_6B/code/physcore_runs_exp2 \
  --folds 5 \
  --epochs 60 \
  --batch_size 128 \
  --num_workers 12 \
  --video_frames 32 \
  --cond_video_frames 32 \
  --lr 3e-4 \
  --weight_decay 1e-3 \
  --force_recache







python /data/model/InternVideo2-Stage2_6B/code/physcore_train_infer.py infer \
  --data_root /data/sunny/real_data \
  --model_dir /data/model/InternVideo2-Stage2_6B/code/physcore_runs_exp2 \
  --submission_zip '/data/sunny/submission_real.zip' \
  --batch_size 256 \
  --num_workers 12 \
  --video_frames 32 \
  --cond_video_frames 32





0.38第一版推理 data/model/InternVideo-main/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py

0.44第一次训练data/model/InternVideo-main/InternVideo2/multi_modality/demo/train_quality_head_lovif_first_strict.py 这个文件会用到data/model/InternVideo-main/InternVideo2/multi_modality/demo/train_quality_head.py

1b模型位置data/model/OpenGVLab/InternVideo2-Stage2_1B-224p-f4

最好的提交结果文件data/sunny/LoViF_record/submitt/1b_strict_train_only_0.44

官方数据集data/sunny/real_data

data/sunny/submission_out/submission_ready.zip 官方提交模板





data/model/InternVideo2-Stage2_6B 一个6b模型位置

data/model/InternVideo2-Stage2_6B 另一个











本周进展：

本周在 LoViF/PhyScore 竞赛上，主要围绕 **InternVideo 系列特征 + 轻量质量头/奖励头** 做了持续迭代与验证：先基于官方数据集 `real_data` 梳理清楚了训练标注表、`train/test/condition` 目录和官方提交模板的对应关系，明确了任务是同时预测 **4 个评分维度**（视频质量、物理真实性、条件对齐、时空一致性）以及 **物理异常时间戳区间**；随后先后尝试了 **1B 与 6B 两条路线**，其中 1B 严格训练版曾做到 **0.44**，而早期直接推理版约 **0.38**。在 6B 方向上，重点跑通并分析了 `physcore_train_infer.py` 这套 **手工视频/条件特征 + MLP 多任务头 + 5 折训练/集成** 的 baseline，实测 5 折训练能够稳定收敛，但从日志看大多在前 10 到 20 个 epoch 快速下降、后续进入平台，但效果没有提升。还尝试评估了几类增强思路，包括 **多视角时间增强、伪样本时间采样扩增、时间戳 mask/soft boundary 监督增强**，前两类实验暂未带来提升，但也帮助确认了“单纯靠采样扩增不够，后续更值得往更强多模态表征、条件信息利用、GroupKFold、异常头重构和输出后处理优化”这个方向继续推进。

最初使用 **InternVideo2-1B** 配合简单的 MLP 预测头，随后升级到参数量更大的 **InternVideo2-6B** 模型并引入了特征缓存机制以加速迭代；为了对齐官方 **SRCC/PLCC** 评价指标，你引入了**可微皮尔逊损失（Pearson Loss）并利用 CLIP提取文本/图像条件特征实现多模态融合；针对 0.36 分的瓶颈，你进一步实施了\**视频总时长归一化（Duration Normalization）**、**分层 K 折交叉验证（Stratified K-Fold）\**以及\**异常概率门槛网格搜索（Threshold Grid Search）**，构建了一个具备独立多任务分支的进阶版 MLP 集成架构。





### 数据大小指标

  idx   width   height  fps     frames  video_path
  1     512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2950.mp4
  2     1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/2202.mp4
  3     1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/2180.mp4
  4     512     320     10.000  16      /data/sunny/real_data/train/image-4D/CamI2V/a living room with hardwood floors and a white couch.mp4
  5     512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/3282.mp4
  6     2560    1920    30.000  1800    /data/sunny/real_data/condition/video-4D_output_uniform_traj_egocentric_20250909_0112_score_1.0000_3fW52QAjYTv_gen.mp4
  7     1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/1783.mp4
  8     1920    1080    29.970  304     /data/sunny/real_data/condition/video-4D_output_uniform_ex4d_high_motion_e0583acd8a9de028629dc86073c2a9ecd3ef9e45f07fefe5e73e219d05bf7603_output.mp4
  9     1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/1635.mp4
  10    512     320     15.000  61      /data/sunny/real_data/test/text-2D/hunyuan_xedit/2499.mp4
  11    1280    704     24.000  121     /data/sunny/real_data/test/text-2D/cosmos_challenging/1801.mp4
  12    640     360     25.000  2495    /data/sunny/real_data/condition/video-4D_output_uniform_vista_non-physical_snowy_score_0.7441_b4fPvWaX5uA.mp4
  13    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2725.mp4
  14    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2888.mp4
  15    720     480     8.000   49      /data/sunny/real_data/test/text-2D/cogvideoX5b_hard_upsampled/216.mp4
  16    480     360     30.006  5184    /data/sunny/real_data/condition/video-4D_output_uniform_recam_non-physical_night_score_0.4828_WsEuZlmEOvF.mp4
  17    1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/1357.mp4
  18    1920    1080    29.953  225     /data/sunny/real_data/condition/video-4D_output_uniform_vista_dynamics_high_motion_0ba50e8a7dbba5a2ea95d9a75d997eb157968d4c87663b70bfb3b38af8305802.mp4
  19    720     480     8.000   49      /data/sunny/real_data/train/text-2D/cogvideoX5b_hard_upsampled/442.mp4
  20    720     480     8.000   49      /data/sunny/real_data/test/text-2D/cogvideoX5b_hard_upsampled/1149.mp4
  21    1280    704     24.000  121     /data/sunny/real_data/test/text-2D/cosmos_challenging/1951.mp4
  22    720     480     8.000   49      /data/sunny/real_data/train/image-4D/Diffusion_as_Shader/6bc60e956b96a11fa4761547b9da9525c5d8c438678b2faea04a4f9518da43f9_middle_frame.mp4
  23    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2616.mp4
  24    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2687.mp4
  25    720     480     8.000   49      /data/sunny/real_data/train/image-4D/Diffusion_as_Shader/a large waterfall in the middle of a building.mp4
  26    4096    2160    25.000  466     /data/sunny/real_data/condition/video-4D_output_uniform_recam_non-physical_shaky_score_0.4216_1J3W4f4gplY.mp4
  27    832     480     30.000  81      /data/sunny/real_data/train/video-4D/output_uniform_recam/optics_high_motion_fa3e201bc3f28a3e33fd2af1dfe0f09f0714351c837aea00671cfa47e2499abf.mp4
  28    480     360     30.000  1656    /data/sunny/real_data/condition/video-4D_output_uniform_recam_non-physical_egocentric_score_0.2483_03B4AIYGwJi.mp4
  29    720     480     8.000   49      /data/sunny/real_data/train/text-2D/cogvideoX5b_hard_upsampled/710.mp4
  30    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/3038.mp4
  31    720     480     8.000   49      /data/sunny/real_data/train/text-2D/cogvideoX5b_hard_upsampled/359.mp4
  32    1280    704     24.000  121     /data/sunny/real_data/test/text-2D/cosmos_challenging/2114.mp4
  33    512     320     15.000  61      /data/sunny/real_data/test/text-2D/hunyuan_xedit/2293.mp4
  34    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2421.mp4
  35    1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/1246.mp4
  36    720     480     8.000   49      /data/sunny/real_data/train/image-4D/Diffusion_as_Shader/A black and white abstract video featuring mesmerizing bubbles.mp4
  37    832     480     30.000  81      /data/sunny/real_data/train/video-4D/output_uniform_recam/thermodynamics_high_motion_3ab9009ced44d0d5b15a3953b9b83df24695fbf5c5aa6781e4d53f43a7c080fc.mp4
  38    3840    2160    25.000  250     /data/sunny/real_data/condition/video-4D_output_uniform_vista_optics_low_motion_8f39469ce3aabc2dae7da9bf3e6197707603d1a2e06a03fbd7c63d0a04b4fce3.mp4
  39    720     480     8.000   49      /data/sunny/real_data/test/text-2D/cogvideoX5b_hard_upsampled/174.mp4
  40    672     384     10.000  49      /data/sunny/real_data/train/video-4D/output_uniform_traj/foggy_20250909_0100_score_0.2960_S4FGAOruVGq_gen.mp4
  41    832     480     30.000  81      /data/sunny/real_data/train/video-4D/output_uniform_recam/non-physical_virtual_score_0.5236_2gnNorQZpib.mp4
  42    672     384     10.000  49      /data/sunny/real_data/test/video-4D/output_uniform_traj/movie_20250909_1002_score_0.5328_7YIcTkvq9rw_gen.mp4
  43    640     360     24.000  444     /data/sunny/real_data/condition/video-4D_output_uniform_vista_non-physical_cartoon_score_0.3442_nSHK4PfNsFj.mp4
  44    3840    2160    29.970  210     /data/sunny/real_data/condition/video-4D_output_uniform_traj_high_motion_20250909_1052_1fd29e58446ceef5b613b4ee45cf5ae1bc7370eb80bdbb9bce7a3e36576c949e_gen.mp4
  45    640     360     25.000  3762    /data/sunny/real_data/condition/video-4D_output_uniform_traj_rainy_20250909_0941_score_0.2860_WKN98wP8Q4k_gen.mp4
  46    512     320     15.000  61      /data/sunny/real_data/train/text-2D/hunyuan_xedit/2612.mp4
  47    1280    704     24.000  121     /data/sunny/real_data/train/text-2D/cosmos_challenging/1450.mp4
  48    512     320     10.000  16      /data/sunny/real_data/train/image-4D/CamI2V/3ab9009ced44d0d5b15a3953b9b83df24695fbf5c5aa6781e4d53f43a7c080fc_middle_frame.mp4
  49    512     320     10.000  16      /data/sunny/real_data/test/image-4D/CamI2V/A teddy bear is climbing over a wooden fence.mp4
  50    640     360     25.000  3762    /data/sunny/real_data/condition/video-4D_output_uniform_recam_non-physical_rainy_score_0.2860_WKN98wP8Q4k.mp4





### 参数qwen

• 要改，但只建议小改 2-3 个参数，不用改模型结构。

  结论基于你这 50 个样本：

  - 主体分辨率是 512x320 / 720x480 / 1280x704
  - 帧数很多是 49/61/121，也有长视频（几百到几千帧）
  - 你现在 --num-video-frames 4 --video-size 192 偏“极速”，对细节和时序有点吃亏

  建议：

  1. 如果你要“半小时内出结果”
     --epochs 10 --batch-size 4 --grad-accum 1 --num-video-frames 4 --video-size 192
  2. 如果你要“分数更稳”
     --epochs 12~20 --batch-size 2 --grad-accum 2 --num-video-frames 8 --video-size 224

  额外关键点：

  - 你采样里很多 condition/ 目录的 4K 长视频，但当前脚本训练/推理读的是 train_root/test_root，这些通常不直接参与，不必专门为 4K 调参。
  - 你脚本里 --video-fps、--max-pixels 目前基本不生效（现在是 cv2 固定抽帧路径），真正影响的是 num-video-frames 和 video-size。

  如果你同意，我可以直接把当前运行切到“epochs=10 快速出包”配置。





打的是音频赛道，但我的模型输入是文本，导致效果不是很好



改动： 我现在要落两类改动：一类是本地 real_data 音频转写/适配脚本，另一类是下载一个公开的
  更强 MeViS Swin-B 模型仓库和 checkpoint 作为主推理方案。先把外部仓库拉下来确认结
  构。



    1. 落两个辅助脚本：音频转写脚本、Swin-Small checkpoint 组合脚本。
    2. 生成新的 Swin-Small 推理配置，并把 real_data 音频 metadata 改成可直接喂模型的文
       本表达。
    3. 做一次单卡 smoke test，确认后就能直接切四卡正式跑。







使用{"session_id":"019cdc10-d0cd-7822-bd79-cf4cab53a176","ts":1773221484,"text":"你好"}

codex resume 019cdc9f-cd44-7ce0-a778-f60d093a4c3d试试到时候

019cdc9f-cd44-7ce0-a778-f60d093a4c3d

019cdc9f-cd44-7ce0-a778-f60d093a4c3d

你的目标很明确了，当前 0.2425 的 J&F 主要先靠三件事抬起来：先把赛道链路改对，其次把视觉主干从 Tiny 升到 Small，再把推理尺度和长视频 chunk 参数拉到更合理的位置。刚才 ASR 没跑通不是模型问题，而是环境
  里有一个坏掉的代理变量 http://127.0.0.1:17897export，我先把这个坑补掉，再按你的 4 x 2080 Ti 配置把默认推理并行改成 4 卡。







• Swin-Small 是 Swin Transformer Small，比你现在这套 Swin-Tiny 更大的视觉主干。

  这次我给你用它，不是随便换个名字，而是因为它在当前工程里是最稳的升级路线：

  - Tiny 现在是你的明显短板，表征容量不够
  - Small 比 Tiny 更深，时序 RVOS 里通常对 J 和 F 都更有帮助
  - 它和当前工程的通道维度兼容性比较好，所以我可以保住现有 MeViS 任务头权重，只替换成更强的视觉 backbone
  - 相比直接硬切 Swin-B/L，Small 在你这台 4 x 2080 Ti 上更现实，推理更容易先跑通

  在你这个仓库里，两者的关键差别主要是 stage depth：

  - Swin-Tiny: DEPTHS = [2, 2, 6, 2]
  - Swin-Small: DEPTHS = [2, 2, 18, 2]

  也就是第三个 stage 更深，视觉特征更强，但整体结构没变，所以适合做“只推理、不训练”的务实升级。

  你现在这次跑的不是官方完整 MeViS Swin-Small 微调成品，而是：

  - 用现有 MeViS_Swin_Tiny.pth 里的任务头
  - 拼上官方 Swin-Small 预训练 backbone
  - 再跑音频转文本后的推理

  对应组合 checkpoint 路径是：

  - /data/sunny/mevis/MeViS_github/.cache/MeViS_Swin_Small_fromTinyHead.pth

  如果你后面能拿到真正的 MeViS 上微调过的 Swin-B/L 或 Swin-Small 成品权重，那会比现在这个“拼接版”更强。







  当前状态：

  - 训练数据已经补齐并可用，根目录是 /data/sunny/mevis/mevis_data
  - 我已经加好了“全局小目标优化”训练改动
  - 训练 smoke test 已经重新启动，当前正在正常初始化训练流程
  - 上一次 smoke test 的崩溃点 np.int 已经修掉了

  这次训练我实际改进去的核心参数是：

  - 更强 backbone：Swin-Small
  - 更高训练分辨率：MIN_SIZE_TRAIN=(512...768), MAX_SIZE_TRAIN=1280
  - 更高测试分辨率：MIN_SIZE_TEST=768, MAX_SIZE_TEST=1280
  - small-object oversample：INPUT.SMALL_OBJECT_AUG.ENABLED=True
  - small-object oversample prob：0.7
  - small-object area threshold：1024.0
  - small-object loss reweight：MODEL.VITA.SMALL_OBJECT.ENABLED=True
  - area ratio：0.01
  - max weight：2.5
  - 更激进的小目标推理阈值：TEST_OUTPUT_THRESHOLD=0.6
  - chunk 调整：TEST_RUN_CHUNK_SIZE=10

  训练 config 在这里：

  - lmpm_SWIN_S_small_object_train.yaml

  代码改动主要在这里：

  - config.py
  - dataset_mapper.py
  - lmpm_model.py
  - augmentation.py

  等这个 smoke test 完成后，我就直接切正式训练，然后训练完把最终 checkpoint 参数和推理命令一起给你，并直接接着推理。





### 上下文

• 这次没有在前向第一步炸掉，说明方向对了。它还在做那套很重的 dataset 序列化初始化，等这一轮 2 iter 结束后我就直接切 5000 iter。

• 5000 iter 已经在执行前的最后验证阶段了，但我还没正式启动那一轮，因为我必须先让这次 2 iter smoke 跑完，确认不会再次 OOM 或在 iteration 里炸掉。

  当前状态：

  - 旧版高分辨率配置会在 2080 Ti 上 OOM
  - 我已经切到适配 4 x 2080 Ti 的训练版配置：
    - Swin gradient checkpoint 开启
    - SAMPLING_FRAME_NUM: 4
    - MIN_SIZE_TRAIN: 448-640
    - MAX_SIZE_TRAIN: 1024
    - CROP.SIZE: (448, 768)
    - 仍保留 small-object oversample 和 loss reweight
    - MAX_ITER: 5000
  - 这版 smoke 目前没有重现上一轮的 OOM，正在完成重型初始化并进入训练验证

  正式 5000 iter 会用这个配置：

  一旦 smoke 结束通过，我会直接启动：

  - 4 卡
  - 5000 iter
  - 训练结束后直接推理

  如果你现在只想要一句结论：5000 iter 我已经切到可跑方案了，差最后一个 smoke 验证通过就正式开训。







• 我的短训策略是“先验证收益，再决定是否续训”，不是一口气长跑。

  具体是这套：

  第一阶段：最小可交付训练

  - 跑 3000 iter
  - 目标不是收敛，而是尽快把小目标改动真正灌进权重
  - 训练完立刻推理，产出一版可提交结果

  为什么是 3000

  - 足够让新采样策略、small-object loss 和更高分辨率开始起作用
  - 训练时长相对可控
  - 就算收益一般，也不会把你两天时间全吃掉

  第二阶段：快速判断是否值得续训
  训练完不是盲目继续，而是看：

  - 本地验证 J&F 是否提升
  - 小目标视频是否明显更稳
  - 推理结果是否比当前版本更像“能交的版本”

  如果提升明显：

  - 续训到 5000 iter

  - 再续到 8000 iter

  如果提升不明显：

  - 立刻停训，转去做推理侧优化和集成，不再浪费时间

  这套短训真正优化的是什么

  - 提高小目标被采到的概率
  - 提高小目标在 loss 中的权重
  - 让更高分辨率真正进入训练，而不是只在推理时硬拉尺度
  - 尽快把“鸽子/远处小人/细小运动目标”这类短板补一点上来

  训练后的动作
  每一轮短训结束后我都会直接做三件事：

  - 给你 checkpoint 路径
  - 给你这轮实际训练参数
  - 直接开始推理，不让显卡空转

### 新对话

• 新对话里直接发这段：

  继续我们刚才的 MeViS 竞赛训练任务。

  工作目录：
  /data/sunny/mevis/MeViS_github

  训练数据真实路径：
  /data/sunny/mevis/mevis_data

  训练用数据根：
  /data/sunny/mevis/training_datasets
  其中 mevis 已经整理成代码可读结构。

  当前目标：

    1. 先完成 smoke test
    2. smoke 通过后立刻启动 4 卡 5000 iter 正式训练
    3. 训练结束后立刻开始推理
    4. 推理结果放到单独新目录
    5. 输出训练参数、checkpoint 路径、推理输出路径

  当前使用的正式训练配置：
  /data/sunny/mevis/MeViS_github/configs/lmpm_SWIN_S_small_object_train_2080ti.yaml

  关键改动文件：
  /data/sunny/mevis/MeViS_github/lmpm/config.py
  /data/sunny/mevis/MeViS_github/lmpm/data/dataset_mapper.py
  /data/sunny/mevis/MeViS_github/lmpm/lmpm_model.py
  /data/sunny/mevis/MeViS_github/lmpm/data/augmentation.py

  当前 smoke 输出目录：
  /data/sunny/mevis/MeViS_github/output/smoke_small_object_train_2080ti_v2

  当前 checkpoint：
  /data/sunny/mevis/MeViS_github/.cache/MeViS_Swin_Small_fromTinyHead.pth

  要求：
  少说话，直接继续检查 smoke 是否通过；通过后马上启动 4 卡 5000 iter 训练，并在训练结束后直接推理。

  如果你想更稳一点，再补一句：

  先检查是否还有 train_net_lmpm.py 在跑，避免重复起任务。



### 训练

• 训练后会生成：

  - checkpoint 目录
    /data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1
  - 最终模型
    /data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1/
    model_final.pth
  - 训练日志
    /data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1/
    log.txt
  - 配置文件
    /data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1/
    config.yaml









  - /data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1

  关键文件：

  - 最终模型：/data/sunny/mevis/MeViS_github/output/
    mevis_swin_small_small_object_train_2080ti_run1/model_final.pth
  - 末轮模型：/data/sunny/mevis/MeViS_github/output/
    mevis_swin_small_small_object_train_2080ti_run1/model_0004999.pth
  - 日志：/data/sunny/mevis/MeViS_github/output/
    mevis_swin_small_small_object_train_2080ti_run1/log.txt
  - 配置：/data/sunny/mevis/MeViS_github/output/
    mevis_swin_small_small_object_train_2080ti_run1/config.yaml

  这个目录里已经有 inference/，说明训练后推理也已经开始写结果了。









• 现在这次推理用的是你刚训练完的这套模型：

  - 配置：/data/sunny/mevis/MeViS_github/configs/lmpm_SWIN_S_small_object_train_2080ti.yaml
  - 权重：/data/sunny/mevis/MeViS_github/output/mevis_swin_small_small_object_train_2080ti_run1/model_final.pth

  模型是什么
  它本质上还是 LMPM / VITA 这条时序 RVOS 框架，只是我做了两类改动：

  - backbone 从原来的 Swin-Tiny 提到了 Swin-Small
  - 训练时加入了面向小目标的改动

  核心组成是：

  - 视觉主干：Swin Transformer Small
  - 文本编码器：RoBERTa
  - 分割头：Mask2Former 风格的 mask decoder
  - 时序模块：VITA，负责跨帧建模和视频级 mask 预测

  这次训练后模型的关键设置
  和默认 baseline 相比，当前实际在用的关键参数是：

  - Swin-Small
  - NUM_OBJECT_QUERIES = 8
  - SAMPLING_FRAME_NUM = 3
  - SAMPLING_FRAME_RANGE = 4
  - 小目标增强与重加权开启
  - 这是为了在 4x2080Ti 上稳定训练并尽量保住小目标效果

  输入是什么
  这次你跑的是 real_data 音频赛道，但当前模型本体仍然是“文本驱动”的 RVOS 模型，所以我走的是：

    1. real_data 里的音频文件
    2. 先通过 ASR 转成文本表达
    3. 每个样本的输入变成：
       - 一段视频帧序列
       - 一条由音频转写得到的文本表达

  也就是说，模型真正吃进去的是：

  - 视频帧：real_data/runtime_datasets_audio/mevis/valid/JPEGImages/...
  - 文本表达：real_data/runtime_datasets_audio/mevis/valid/meta_expressions.json 里的 exp

  不是直接把音频波形送进模型，而是“音频 -> 文本 -> 文本 RVOS 模型”。

  输出是什么
  模型输出的是每个表达在每一帧上的二值分割 mask。

  落盘形式是：

  - 每帧一个 .png
  - 目录结构：
    /data/sunny/mevis/MeViS_github/output/mevis_audio_realdata_from_train_run1_4gpu/inference/<video_id>/<exp_id>/
    *.png

  每张 png 表示：

  - 白色或前景值：目标区域
  - 黑色或背景值：非目标区域

  最后这些 png 会再打包成提交需要的 zip。

  一句话概括
  这次推理模型是：

  - Swin-Small + RoBERTa + Mask2Former/VITA
  - 用 MeViS 训练集做了 5000 iter 微调
  - 输入是“视频帧 + 由音频转写得到的文本”
  - 输出是“每个表达在每一帧上的分割 mask PNG”



















经常安装的配置：









• 这个模型确实复杂一些，本质是“多路特征融合回归”：

    1. RQ-VQA 主干

  - 输入不是只有视频帧，而是 4 类特征一起回归分数。
  - 最终用官方 KVQ 训练好的 10 个 checkpoint 做集成平均（不是你现在重新训练出来的权重）。

    2. 四类特征来源

  - SlowFast：视频时序/运动特征（每个视频取 8 帧，对应 8 个时刻特征）。
  - LIQE：图像质量语义特征（逐帧抽）。
  - Q-Align：大模型质量感知特征（逐帧抽）。
  - FAST-VQA：时空质量特征（视频级一个向量，后面复制到 8 帧维度对齐）。

    3. 你问的“数据拆分”是什么意思

  - 这里的 part1/part2 不是训练集/验证集拆分。
  - 是把同一个测试集列表拆成两半，让两张 3090 并行提特征，加速推理。
  - 最后会合并成完整测试集结果，不会改变样本顺序和内容。


  当前我这边正处在第一步：SlowFast 在下载官方预训练权重（264MB），下完就开始双卡抽取特征。



  四路“权重/模型”的意义：

    1. SlowFast 权重：提取运动时序特征（视频动作、动态稳定性）。
    2. LIQE.pt：提取逐帧图像质量特征（噪声、模糊、压缩感等）。
    3. Q-Align(one-align)：提取大模型质量语义特征（更偏主观质量理解）。
    4. FAST-VQA 权重：提取时空质量特征（兼顾帧内与帧间）。

  然后再走一次：

    5. KVQ test ckpts（10个）：把上面四路特征融合，输出最终 VideoQuality 分数（10个模型做平均更稳）。

  所以不是“同一个模型跑四遍”，而是四个不同能力的特征提取器，各司其职，最后由 RQ-VQA 头部融合。









观察训练集标签，似乎1 和 10出现的概率很少，让模型输出这俩个数字的概率少一些