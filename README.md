### pip3安装

1 onnx: 1.7.0  
2 onnx-tf: 1.5.0  
3 目前228机器环境可用

-----------
### 说明
**1 pytorch 模型转 tensorflow pb 模型**  
执行pytorch_2_tensorFlow.py 的 pytorch2tensorflow函数.

**2 测试tensorflow pb 模型**  
执行pytorch_2_tensorFlow.py 的 test_tensorflow函数, 需要知道输入/输出节点名称.

**3 tensorflow pb 转 tensorflow server pb**  
执行 tfpb_2_tfserverpb.py 的 restore_pb_and_save_serverpb 函数, 需要知道输入/输出节点名称.

**4 部署tensorflow server 模型， 只能用gpu部署，模型输入NCWH，cpu不支持（目前在228转模型，在226部署测试）**    
例如： docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
-idt --rm -p 8511:8500 --mount type=bind,source=/data/tiantian/model_convert/model_path,target=/models/siamese_tfserving_graph \
-e MODEL_NAME=siamese_tfserving_graph -t tensorflow/serving:1.14.0-gpu \
--per_process_gpu_memory_fraction=0.1

**5 测试 tensorflow server模型**    
执行 tfpb_2_tfserverpb.py 的 test_pb 函数.
