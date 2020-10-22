# -*- coding: utf-8 -*-
"""
Created on 2020-10-21 11:47

@author: tiantian
"""

import os
import cv2
import numpy as np

# model
from res18 import ResNet18SiameseNetwork

#onnx - step 1
from torch.autograd import Variable
import torch.onnx
from onnx_tf.backend import prepare

# onnx - step 2
import onnx
import onnxruntime as ort

#
# # tensorflow - step 3
import tensorflow as tf

print(tf.__version__)

WIDTH = 224
HEIGHT = 224
test_img = cv2.imread("1.jpg")

def test_pytorch(model, image):
    img_resized = cv2.resize(image, (WIDTH, HEIGHT)).astype(np.float32)
    img_tensor = torch.Tensor(np.expand_dims(np.transpose(img_resized, (2, 0, 1)), axis=0))
    feature = model(img_tensor)
    ndarray_vector = feature.detach().cpu().numpy()
    list_vector = ndarray_vector.tolist()
    list_vector = list_vector[0]
    return list_vector

def test_onnx(model_path, image):
    img_resized = cv2.resize(image, (WIDTH, HEIGHT)).astype(np.float32)
    img_resized = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), axis=0)
    sess = ort.InferenceSession(model_path)
    feature = sess.run(None, {'input.1': img_resized})[0][0]
    return feature

def test_tensorflow(tf_pb_path, input_name='input.1:0', output_name='198:0'):
    """
    :param tf_pb_path: model path
    :param input_name: the input name of model
    :param output_name: the output name of model,  "https://lutzroeder.github.io/netron/" can check input and output name
    :return:
    """
    with tf.Session() as sess:
        with tf.gfile.GFile(tf_pb_path, 'rb') as f:  # 加载模型
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

            input = sess.graph.get_tensor_by_name(input_name)
            output = sess.graph.get_tensor_by_name(output_name)
            img_resized = cv2.resize(test_img, (WIDTH, HEIGHT)).astype(np.float32)
            img_resized = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), axis=0)
            output_tf_pb = sess.run([output], feed_dict={input: img_resized})[0][0]

            #print('output_tf_pb = {}'.format(output_tf_pb))
            print(output_tf_pb.shape)
            print(output_tf_pb[:10])
            return output_tf_pb

def compare_feature(feat1, feat2):
    diff = np.abs(feat1-feat2)
    diff = np.sum(diff)
    print('compare feature, the diff is', diff)

def pytorch2tensorflow():

    model_path = "model_path"

    # step 1 init pytorch
    cuda_ = 'cuda:{}'.format(0)
    device = torch.device(cuda_ if torch.cuda.is_available() else 'cpu')
    pytorch_model_path = os.path.join(model_path, "siamese.ckpt")
    model = ResNet18SiameseNetwork()
    model_params = torch.load(pytorch_model_path)
    siamese_state_dict = {}
    for k, v in model_params.items():
        siamese_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(siamese_state_dict)
    model.eval()

    # step 2 pytorch to onnx
    dummy_input = Variable(torch.randn(1, 3, HEIGHT, WIDTH))  # nchw
    onnx_model_path = os.path.join(model_path, "siamese.onnx")
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)
    pytorch_feature = test_pytorch(model, test_img)
    onnx_feature = test_onnx(onnx_model_path, test_img)
    compare_feature(pytorch_feature, onnx_feature)
    print("pytorch to onnx ok!")
#
    # step 3 onnx --> tf pb
    onnx_model = onnx.load(onnx_model_path)
    tf_pb_path = os.path.join(model_path, "siamese.pb")
    print("1")
    tf_rep = prepare(onnx_model, strict=False)
    print("2")
    tf_rep.export_graph(tf_pb_path)
    #tf_feature = test_tensorflow(tf_pb_path)
    #compare_feature(pytorch_feature, tf_feature)
    print("onnx to tensorflow ok!")

#pytorch2tensorflow()

test_tensorflow(os.path.join('./model_path', "siamese.pb"))


