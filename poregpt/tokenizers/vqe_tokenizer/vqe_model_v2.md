"""
2026-02-28T19:23:09.448144425+08:00 i-d070lc6heob1qf675c60 预训练模型权重键 (前几个): ['encoder.0.mean_conv.weight', 'encoder.0.std_conv.weight', 'encoder.1.weight', 'encoder.1.bias', 'encoder.1.running_mean']
2026-02-28T19:23:09.449206069+08:00 i-d070lc6heob1qf675c60 当前模型权重键 (前几个): ['cnn_model.encoder.0.mean_conv.weight', 'cnn_model.encoder.0.std_conv.weight', 'cnn_model.encoder.1.weight', 'cnn_model.encoder.1.bias', 'cnn_model.encoder.1.running_mean']
2026-02-28T19:23:09.449839168+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.0.mean_conv.weight
2026-02-28T19:23:09.450044383+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.0.std_conv.weight
2026-02-28T19:23:09.450238497+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.1.weight
2026-02-28T19:23:09.450432992+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.1.bias
2026-02-28T19:23:09.450630969+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.1.running_mean
2026-02-28T19:23:09.450823905+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.1.running_var
2026-02-28T19:23:09.451014678+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.1.num_batches_tracked
2026-02-28T19:23:09.451192236+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.3.weight
2026-02-28T19:23:09.451378961+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.4.weight
2026-02-28T19:23:09.451574871+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.4.bias
2026-02-28T19:23:09.451767731+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.4.running_mean
2026-02-28T19:23:09.451949055+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.4.running_var
2026-02-28T19:23:09.452133258+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.4.num_batches_tracked
2026-02-28T19:23:09.452318463+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.6.weight
2026-02-28T19:23:09.452502607+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.7.weight
2026-02-28T19:23:09.452709054+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.7.bias
2026-02-28T19:23:09.452885013+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.7.running_mean
2026-02-28T19:23:09.453078996+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.7.running_var
2026-02-28T19:23:09.453256629+08:00 i-d070lc6heob1qf675c60 加载参数:cnn_model.encoder.7.num_batches_tracked
2026-02-28T19:23:09.456067254+08:00 i-d070lc6heob1qf675c60 ✅ 加载了 19 个encoder参数
2026-02-28T19:23:09.456269577+08:00 i-d070lc6heob1qf675c60 🔒 冻结encoder参数
2026-02-28T19:23:09.456489406+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.0.mean_conv.weight
2026-02-28T19:23:09.456696500+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.0.std_conv.weight
2026-02-28T19:23:09.456890972+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.1.weight
2026-02-28T19:23:09.457080215+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.1.bias
2026-02-28T19:23:09.457277400+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.3.weight
2026-02-28T19:23:09.457471936+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.4.weight
2026-02-28T19:23:09.457666882+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.4.bias
2026-02-28T19:23:09.457861969+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.6.weight
2026-02-28T19:23:09.458052779+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.7.weight
2026-02-28T19:23:09.458242905+08:00 i-d070lc6heob1qf675c60 冻结参数:cnn_model.encoder.7.bias
2026-02-28T19:23:09.458635694+08:00 i-d070lc6heob1qf675c60 ✅ 冻结了 10 个encoder参数

这段详细的日志非常清晰地展示了整个加载和冻结的过程，完美地证实了我们的分析。

加载过程分析 (Load Process):

映射成功： 日志显示，预训练权重 encoder.xxxx 已被成功映射并加载到 cnn_model.encoder.xxxx。
加载条目明细：
    cnn_model.encoder.0.mean_conv.weight (卷积层参数)
    cnn_model.encoder.0.std_conv.weight (卷积层参数)
    cnn_model.encoder.1.weight (BatchNorm层参数 gamma)
    cnn_model.encoder.1.bias (BatchNorm层参数 beta)
    cnn_model.encoder.1.running_mean (BatchNorm层缓冲区 Buffer)
    cnn_model.encoder.1.running_var (BatchNorm层缓冲区 Buffer)
    cnn_model.encoder.1.num_batches_tracked (BatchNorm层缓冲区 Buffer)
    cnn_model.encoder.3.weight (卷积层参数)
