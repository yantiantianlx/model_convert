# -*- coding: utf-8 -*-
"""
Created on 2020-09-04 11:23

@author: tiantian
"""

import sys, os, io
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import numpy as np
import cv2

def restore_ckpt_and_save_serverpb(input_checkpoint, export_path_base):
    checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print ('name scope: ', graph.get_name_scope())

            export_path_base = export_path_base
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(0)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)


            """ inputs and outpus will be different"""
            inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("Placeholder").outputs[0])
            outputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("resfcn256/Conv2d_transpose_16/Sigmoid").outputs[0])

            labeling_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={"inputs": inputs},
                    outputs={"outputs": outputs},
                    method_name="tensorflow/serving/predict"))

            """
            tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                          输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                          对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                          对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: labeling_signature},
                legacy_init_op=legacy_init_op)
            builder.save()
            print("Build Done")

def restore_pb_and_save_serverpb(export_path_base, model_name='siamese.pb'):
    with tf.gfile.GFile(export_path_base + model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='import')


    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            print('name scope: ', graph.get_name_scope())

            export_path_base = export_path_base
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(0)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            """ inputs and outpus will be different"""
            inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("import/input.1").outputs[0])
            features = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("import/198").outputs[0])

            labeling_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={"inputs": inputs},
                    outputs={"features": features},
                    method_name="tensorflow/serving/predict"))

            """
            tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                          输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                          对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                          对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: labeling_signature},
                legacy_init_op=legacy_init_op)
            builder.save()
            print("Build Done")

"""
docker run \
-idt --rm -p 8511:8500 \
--mount type=bind,source=/data/tiantian/model_convert/model_path,target=/models/siamese_tfserving_graph \
-e MODEL_NAME=siamese_tfserving_graph \
-t tensorflow/serving:1.14.0-gpu

"""

"""
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
-idt --rm -p 8511:8500 --mount type=bind,source=/data/tiantian/model_convert/model_path,target=/models/siamese_tfserving_graph \
-e MODEL_NAME=siamese_tfserving_graph -t tensorflow/serving:1.14.0-gpu \
--per_process_gpu_memory_fraction=0.1

"""
def test_pb():
    channel = implementations.insecure_channel("127.0.0.1", 8511)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()

    WIDTH = 224
    HEIGHT = 224
    image_np = cv2.imread('./1.jpg')
    image_np = cv2.resize(image_np, (WIDTH, HEIGHT))
    image_np = np.transpose(image_np, (2,0,1)).astype(np.float32)
    request.model_spec.name = 'siamese_tfserving_graph'
    request.inputs['inputs'].CopyFrom( tf.contrib.util.make_tensor_proto(image_np, shape=[1] + list(image_np.shape)))
    result_feature = stub.Predict(request, 1000.)
    predict_body_dict = {}
    for key in result_feature.outputs:
        tensor_proto = result_feature.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        predict_body_dict[key] = nd_array
    features = predict_body_dict['features']
    h, w, _ = image_np.shape
    feature = features[0]
    print(feature[:10])
    print('ok')


# 模型格式转换
#restore_ckpt_and_save_serverpb('./model/alignment', '/data/models/PRNet')
restore_pb_and_save_serverpb('./model_path', 'siamese.pb')

test_pb()

