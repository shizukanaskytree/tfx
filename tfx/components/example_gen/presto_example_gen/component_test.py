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
"""Tests for tfx.components.example_gen.presto_example_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import json_format
from tfx.components.example_gen.presto_example_gen import component
from tfx.components.example_gen.presto_example_gen import presto_config_pb2
from tfx.proto import example_gen_pb2


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()
    presto_config = presto_config_pb2.ConnectionConfig.PrestoConfig(
        host='localhost', port=8080)
    self.conn_config = presto_config_pb2.ConnectionConfig(
        presto_config=presto_config)

  def _extract_conn_config(self, input_config):
    unpacked_input_config = example_gen_pb2.Input()
    json_format.Parse(input_config, unpacked_input_config)

    conn_config = presto_config_pb2.ConnectionConfig()
    unpacked_input_config.custom_config.Unpack(conn_config)
    return conn_config

  def test_construct(self):
    presto_example_gen = component.PrestoExampleGen(
        self.conn_config, query='query')
    self.assertEqual(
        self.conn_config,
        self._extract_conn_config(
            presto_example_gen.exec_properties['input_config']))
    self.assertEqual('ExamplesPath',
                     presto_example_gen.outputs.examples.type_name)
    artifact_collection = presto_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_with_output_config(self):
    presto_example_gen = component.PrestoExampleGen(
        self.conn_config,
        query='query',
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
                example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
            ])))
    self.assertEqual(
        self.conn_config,
        self._extract_conn_config(
            presto_example_gen.exec_properties['input_config']))
    self.assertEqual('ExamplesPath',
                     presto_example_gen.outputs.examples.type_name)
    artifact_collection = presto_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)
    self.assertEqual('test', artifact_collection[2].split)

  def test_construct_with_input_config(self):
    presto_example_gen = component.PrestoExampleGen(
        self.conn_config,
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='query1'),
            example_gen_pb2.Input.Split(name='eval', pattern='query2'),
            example_gen_pb2.Input.Split(name='test', pattern='query3')
        ]))
    self.assertEqual(
        self.conn_config,
        self._extract_conn_config(
            presto_example_gen.exec_properties['input_config']))
    self.assertEqual('ExamplesPath',
                     presto_example_gen.outputs.examples.type_name)
    artifact_collection = presto_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)
    self.assertEqual('test', artifact_collection[2].split)

  def test_bad_construction(self):
    empty_config = presto_config_pb2.ConnectionConfig()
    self.assertRaises(
        RuntimeError,
        component.PrestoExampleGen,
        conn_config=empty_config,
        query='')

    port_only_config = presto_config_pb2.ConnectionConfig()
    port_only_config.presto_config.port = 8080
    self.assertRaises(
        RuntimeError,
        component.PrestoExampleGen,
        conn_config=port_only_config,
        query='')


if __name__ == '__main__':
  tf.test.main()
