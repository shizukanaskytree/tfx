# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.components.example_gen.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import any_pb2
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2


class UtilsTest(tf.test.TestCase):

  def test_dict_to_example(self):
    instance_dict = {
        'int': 10,
        'float': 5.0,
        'str': 'abc',
        'int_list': [1, 2],
        'float_list': [3.0],
        'str_list': ['ab', 'cd'],
        'none': None,
        'empty_list': [],
    }
    example = utils.dict_to_example(instance_dict)
    self.assertProtoEquals(
        """
        features {
          feature {
            key: "empty_list"
            value {
            }
          }
          feature {
            key: "float"
            value {
              float_list {
                value: 5.0
              }
            }
          }
          feature {
            key: "float_list"
            value {
              float_list {
                value: 3.0
              }
            }
          }
          feature {
            key: "int"
            value {
              int64_list {
                value: 10
              }
            }
          }
          feature {
            key: "int_list"
            value {
              int64_list {
                value: 1
                value: 2
              }
            }
          }
          feature {
            key: "none"
            value {
            }
          }
          feature {
            key: "str"
            value {
              bytes_list {
                value: "abc"
              }
            }
          }
          feature {
            key: "str_list"
            value {
              bytes_list {
                value: "ab"
                value: "cd"
              }
            }
          }
        }
        """, example)

  def test_make_default_input_config_without_custom(self):
    input_config = utils.make_default_input_config('query1')
    self.assertEqual(1, len(input_config.splits))
    self.assertEqual(False, input_config.HasField('custom_config'))

  def test_make_default_input_config_with_custom(self):
    custom_config = example_gen_pb2.Input.Split(name='config', pattern='test')
    packed_custom_config = any_pb2.Any()
    packed_custom_config.Pack(custom_config)

    input_config = utils.make_default_input_config(
        split_pattern='query1', custom_config=packed_custom_config)
    self.assertEqual(1, len(input_config.splits))

    # Unpack custom_config
    unpacked_custom_config = example_gen_pb2.Input.Split()
    input_config.custom_config.Unpack(unpacked_custom_config)
    self.assertEqual(custom_config, unpacked_custom_config)

  def test_make_output_split_names(self):
    split_names = utils.generate_output_split_names(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
        ]),
        output_config=example_gen_pb2.Output())
    self.assertListEqual(['train', 'eval'], split_names)

    split_names = utils.generate_output_split_names(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='single/*')
        ]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ])))
    self.assertListEqual(['train', 'eval'], split_names)

  def test_make_default_output_config(self):
    output_config = utils.make_default_output_config(
        utils.make_default_input_config())
    self.assertEqual(2, len(output_config.split_config.splits))

    output_config = utils.make_default_output_config(
        example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
        ]))
    self.assertEqual(0, len(output_config.split_config.splits))


if __name__ == '__main__':
  tf.test.main()
