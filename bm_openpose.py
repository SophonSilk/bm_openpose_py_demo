""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sys
import os
import argparse
import json
import numpy as np
import sophon.sail as sail
import ctypes
import struct
import time
import cv2

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.ab = [x * scale for x in [1, -128, 1, -128, 1, -128]]

  def process(self, input, output, height, width):
    """ Execution function of preprocessing.
    Args:
      input: sail.BMImage, input image
      output: sail.BMImage, output data

    Returns:
      None
    """
    self.bmcv.vpp_resize(input, output,  width, height)

class Net:
  input_shapes_ = {}
  output_shapes_ = {}
  input_tensors_ = {}
  output_tensors_ = {}
  post_process_inputs_ = []
  output_names_ = []
  output_shapes_array_ = []
  preprocessor_ = 0
  tpu_id_ = 0
  handle_ = 0
  img_dtype_ = 0
  engine_ = 0
  graph_name_ = 0
  bmcv_ = 0
  input_name_ = 0
  lib_post_process_ = 0
  input_dtype_ = 0

  def __init__(self, bmodel_path, tpu_id, stage):
    # init Engine
    Net.engine_ = sail.Engine(tpu_id)
    # load bmodel without builtin input and output tensors
    Net.engine_.load(bmodel_path)
    # get model info
    # only one model loaded for this engine
    # only one input tensor and only one output tensor in this graph
    Net.handle_ = Net.engine_.get_handle()
    Net.graph_name_ = Net.engine_.get_graph_names()[0]
    input_names = Net.engine_.get_input_names(Net.graph_name_)
    print("input names:",input_names)
    input_dtype = 0
    Net.tpu_id_ = tpu_id
    Net.input_name_ = input_names[0]
    for i in range(len(input_names)):
      Net.input_shapes_[input_names[i]] = Net.engine_.get_input_shape(Net.graph_name_, input_names[i])
      input_dtype = Net.engine_.get_input_dtype(Net.graph_name_, input_names[i])
      input = sail.Tensor(Net.handle_, Net.input_shapes_[input_names[i]], input_dtype, False, False)
      Net.input_tensors_[input_names[i]] = input
      Net.input_dtype_ = input_dtype
    Net.output_names_ = Net.engine_.get_output_names(Net.graph_name_)
    for i in range(len(Net.output_names_)):
      Net.output_shapes_[Net.output_names_[i]] = Net.engine_.get_output_shape(Net.graph_name_, Net.output_names_[i])
      output_dtype = Net.engine_.get_output_dtype(Net.graph_name_, Net.output_names_[i])
      output = sail.Tensor(Net.handle_, Net.output_shapes_[Net.output_names_[i]], output_dtype, True, True)
      Net.output_tensors_[Net.output_names_[i]] = output
    print ("input shapes:",Net.input_shapes_)
    print ("output shapes:",Net.output_shapes_)

    # set io_mode
    Net.engine_.set_io_mode(Net.graph_name_, sail.IOMode.SYSIO)
    Net.bmcv_ = sail.Bmcv(Net.handle_)
    Net.img_dtype_ = Net.bmcv_.get_bm_image_data_format(input_dtype)
    scale = Net.engine_.get_input_scale(Net.graph_name_, input_names[0])
    print("scale", scale)
    scale *= 0.003922
    Net.preprocessor_ = PreProcessor(Net.bmcv_, scale)

    # load postprocess so
    ll = ctypes.cdll.LoadLibrary
    Net.lib_post_process_ = ll('./post_process_lib/libPostProcess.so')
    Net.lib_post_process_.post_process_hello()

    if os.path.exists('result_imgs') is False:
      os.system('mkdir -p result_imgs')

  def cut(obj, sec):
    return [obj[i : i + sec] for i in range(0, len(obj), sec)]

  def detect(self, video_path):
    print("video_path=", video_path)
    # open a video to be decoded
    decoder = sail.Decoder(video_path, True, Net.tpu_id_)
    frame_id = 0
    cap = cv2.VideoCapture(video_path)
    while 1:
      img = sail.BMImage()
      # decode a frame from video
      ret = decoder.read(Net.handle_, img)
      if ret != 0:
        print("Finished to read the video!");
        return

      # preprocess image for inference
      img_resized = sail.BMImage(Net.handle_, Net.input_shapes_[Net.input_name_][2],
                        Net.input_shapes_[Net.input_name_][3],
                        sail.Format.FORMAT_BGR_PLANAR, img.dtype())
      img_processed = sail.BMImage(Net.handle_, Net.input_shapes_[Net.input_name_][2],
                        Net.input_shapes_[Net.input_name_][3],
                        sail.Format.FORMAT_BGR_PLANAR, Net.img_dtype_)
      # resize origin image
      Net.preprocessor_.process(img,
          img_resized, Net.input_shapes_[Net.input_name_][2], Net.input_shapes_[Net.input_name_][3])

      # split
      Net.bmcv_.convert_to(img_resized, img_processed, ((Net.preprocessor_.ab[0], Net.preprocessor_.ab[1]), \
                                                           (Net.preprocessor_.ab[2], Net.preprocessor_.ab[3]), \
                                                           (Net.preprocessor_.ab[4], Net.preprocessor_.ab[5])))
      Net.bmcv_.bm_image_to_tensor(img_processed, Net.input_tensors_[Net.input_name_])
      # do inference
      #print(Net.input_shapes_)
      Net.engine_.process(Net.graph_name_,
              Net.input_tensors_, Net.input_shapes_, Net.output_tensors_)
      #out_data = Net.output_tensors_[Net.output_names_[0]].pysys_data()
      #print(out_data)

      # post process, nms
      CLONG_P_INPUT = len(Net.output_tensors_) * ctypes.c_long
      Net.post_process_inputs_ = CLONG_P_INPUT()
      for i in range(len(Net.output_tensors_)):
          output_data = Net.output_tensors_[Net.output_names_[i]].pysys_data()
          Net.post_process_inputs_[i] = output_data[0]
      
      ##############################################
      #show image
      dis_img = sail.BMImage(Net.handle_, img.height(),
                        img.width(),
                        sail.Format.FORMAT_BGR_PLANAR, img.dtype())
      Net.bmcv_.vpp_resize(img, dis_img, img.width(), img.height())
      t_img_tensor = sail.Tensor(Net.handle_, [1, 3, img.height(), img.width()], sail.Dtype.BM_UINT8, True, False)
      Net.bmcv_.bm_image_to_tensor(dis_img, t_img_tensor)
      t_img_tensor.sync_d2s()
      np_t_img_tensor = t_img_tensor.asnumpy()
      np_t_img_tensor = np_t_img_tensor.transpose((0, 2, 3, 1))
      np_t_img_tensor = np_t_img_tensor.reshape([img.height(), img.width(), 3])
      ori_img = np.uint8(np_t_img_tensor)
      ori_img = ori_img.ctypes.data_as(ctypes.c_char_p)
      #cv2.imshow('det_result', np.uint8(np_t_img_tensor))
      #cv2.imwrite("tensor.jpg", np.uint8(np_t_img_tensor))
      #cv2.waitKey(10)
      ##############################################

      #print( Net.output_shapes_)
      pointnum = Net.lib_post_process_.post_process(Net.post_process_inputs_, 1,img.width(), img.height(), \
              Net.input_shapes_[Net.input_name_][3], \
              Net.input_shapes_[Net.input_name_][2], \
              Net.output_shapes_[Net.output_names_[0]][3], \
              Net.output_shapes_[Net.output_names_[0]][2], \
              len(Net.output_tensors_), \
              Net.output_shapes_[Net.output_names_[0]][3] * Net.output_shapes_[Net.output_names_[0]][2] * Net.output_shapes_[Net.output_names_[0]][1], \
              ori_img)
      frame_id += 1

      

if __name__ == '__main__':
  """ A openpose example using bm-ffmpeg to decode and bmcv to preprocess.
  """
  parser = argparse.ArgumentParser(description='Process input config file.')
  parser.add_argument('--config', default='', required=True)
  parser.add_argument('--tpu_id', default=0, type=int, required=False)
  args = parser.parse_args()

  with open(args.config,'r') as load_f:
    load_param = json.load(load_f)
    print("bmodel is:", load_param['bmodel_path'])
  print("tpu_id:", args.tpu_id)
  #print("video_info:", load_param['videos'])
  #video_info = load_param['videos']
  #print("path:", load_param['videos'][1]['video_path'])
  openpose_net = Net(load_param['bmodel_path'], args.tpu_id, load_param['bmodel_stage'])
  openpose_net.detect(load_param['videos'][0]['video_path'])
"""
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  openpose_net = Net(ARGS.bmodel, ARGS.tpu_id)
  openpose_net.detect(ARGS.input)
"""

