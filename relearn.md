调api，学思考过程。让他学习思考过程可能会好一些



### team_id

Your officially assigned **Team ID** is: tm_73178fa90b0bd70b





### 思路

• 1. 准备并确认数据
     读取了题目文件 questions.json（JSONL，600题）。
     每行包含 question，并按行号生成 question_id（从 1 到 600）。
  2. 选择并加载本地模型
     使用本地模型目录：/data/model/Qwen/Qwen2-VL-7B-Instruct。
     在脚本里通过 transformers 加载 AutoTokenizer + Qwen2VLForConditionalGeneration，并用 device_map="auto" 让模型自动分配到 GPU。
  3. 编写并使用推理脚本
     实际跑的是 run_qwen_local.py。
     脚本核心逻辑：

  - 逐题读取 question
  - 构造提示词：强制模型“只输出 a 或 b”
  - 用贪心解码生成短输出（max_new_tokens=3, do_sample=False）
  - 正则提取 a/b，如果提取失败回退为 a

  4. 执行全量推理
     执行命令（4卡）：
     CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/sunny/Relearn/run_qwen_local.py --model /data/model/Qwen/Qwen2-VL-7B-Instruct --input /data/model/MMToM-QA-main/Benchmark/questions.json --output /data/sunny/
     Relearn/submission_qwen_local.json
     日志写入 run_qwen_local.log。
     日志显示从 processed=10/600 到 processed=600/600，最终完成。
  5. 生成提交文件
     输出文件是 submission_qwen_local.json。
     格式是官方要求的 JSON 数组：
     {"question_id": 整数, "answer": "a"|"b"}。
  6. 这次用到的文件

  - 输入数据：questions.json
  - 推理代码：run_qwen_local.py
  - 运行日志：run_qwen_local.log
  - 提交结果：submission_qwen_local.json

  补充：这条链路是“本地大模型直接判别 a/b”，没有使用 OpenAI API，也没有使用视频文件。







• Qwen 这版可以理解成 3 个非常直接的步骤：

  1. 喂题目给模型
     我把每道题原文（question）加一句指令一起给模型：
     “你只能回答 a 或 b”。
  2. 模型生成一小段文本
     模型不是直接给程序变量，而是像聊天一样“生成文字”。
     例如它可能生成：

  - a
  - b
  - The answer is b.

  3. 把这段文字转成比赛需要的格式
     比赛只认 "a" 或 "b"。
     所以程序要从模型文字里“读出”到底是 a 还是 b。
     这里的“正则”只是一个文本查找工具：

  - 在输出里找有没有独立的 a 或 b 字符。
  - 找到就用它。
  - 如果完全找不到，就先用默认 a（防止提交文件格式错误）。

  所以不是“神秘处理”，本质就是：
  模型先生成自然语言 -> 程序再把自然语言提取成标准答案 a/b。



