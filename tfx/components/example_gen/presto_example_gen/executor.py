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
"""Generic TFX PrestoExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import prestodb
import tensorflow as tf
from typing import Any, Dict, Iterable, List, Text, Tuple
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen.presto_example_gen import presto_config_pb2
from tfx.proto import example_gen_pb2
from tfx.utils import types
from google.protobuf import json_format


class _ReadPrestoDoFn(beam.DoFn):
  """Beam DoFn class that reads from Presto."""

  def __init__(self, client: prestodb.dbapi.Connection):
    self.cursor = client.cursor()

  def process(self, query: Text):
    """Yields rows from query results."""
    self.cursor.execute(query)
    rows = self.cursor.fetchall()
    if rows:
      cols = []
      col_types = []
      # Returns a list of (column_name, column_type, None, ...)
      # https://github.com/prestodb/presto-python-client/blob/master/prestodb/dbapi.py#L199
      for metadata in self.cursor.description:
        cols.append(metadata[0])
        col_types.append(metadata[1])

      for r in rows:
        yield zip(cols, col_types, r)


def _deserialize_conn_config(
    input_config: example_gen_pb2.Input) -> prestodb.dbapi.Connection:
  """Deserializes input_config to Connection client."""
  config = presto_config_pb2.ConnectionConfig()
  input_config.custom_config.Unpack(config)

  params = {'host': config.presto_config.host}  # Required field
  # Only deserialize rest of parameters if set by user
  if config.presto_config.HasField('port'):
    params['port'] = config.presto_config.port
  if config.presto_config.HasField('user'):
    params['user'] = config.presto_config.user
  if config.presto_config.HasField('source'):
    params['source'] = config.presto_config.source
  if config.presto_config.HasField('catalog'):
    params['catalog'] = config.presto_config.catalog
  if config.presto_config.HasField('schema'):
    params['schema'] = config.presto_config.schema
  if config.presto_config.HasField('http_scheme'):
    params['http_scheme'] = config.presto_config.http_scheme
  if config.presto_config.HasField('auth'):
    params['auth'] = _deserialize_auth_config(config.presto_config.auth)
  if config.presto_config.HasField('max_attempts'):
    params['max_attempts'] = config.presto_config.max_attempts
  if config.presto_config.HasField('request_timeout'):
    params['request_timeout'] = config.presto_config.request_timeout
  if config.presto_config.HasField('isolation_level'):
    params['isolation_level'] = config.presto_config.isolation_level

  return prestodb.dbapi.connect(**params)


def _deserialize_auth_config(
    auth_config: presto_config_pb2.AuthConfig) -> prestodb.auth.Authentication:
  """Deserializes auth config to presto Authentication class."""
  if auth_config.HasField('basic_auth'):
    return prestodb.auth.BasicAuthentication(auth_config.basic_auth.username,
                                             auth_config.basic_auth.password)
  elif auth_config.HasField('kerberos_auth'):
    params = {  # Required fields
        'service_name':
            auth_config.kerberos_auth.service_name,
        'mutual_authentication':
            auth_config.kerberos_auth.mutual_authentication,
        'force_preemptive':
            auth_config.kerberos_auth.force_preemptive,
        'sanitize_mutual_error_response':
            auth_config.kerberos_auth.sanitize_mutual_error_response,
        'delegate':
            auth_config.kerberos_auth.delegate
    }
    # Only deserialize rest of parameters if set by user
    if auth_config.kerberos_auth.HasField('config'):
      params['config'] = auth_config.kerberos_auth.config
    if auth_config.kerberos_auth.HasField('hostname_override'):
      params['hostname_override'] = auth_config.kerberos_auth.hostname_override
    if auth_config.kerberos_auth.HasField('principal'):
      params['principal'] = auth_config.kerberos_auth.principal
    if auth_config.kerberos_auth.HasField('ca_bundle'):
      params['ca_bundle'] = auth_config.kerberos_auth.ca_bundle
    return prestodb.auth.KerberosAuthentication(**params)
  else:
    raise RuntimeError('Authentication type not supported.')


def _row_to_example(
    instance: Iterable[Tuple[Text, Text, Any]]) -> tf.train.Example:
  """Convert presto result row to tf example."""
  feature = {}
  for key, data_type, value in instance:
    if value is None:
      feature[key] = tf.train.Feature()
    elif data_type in {'tinyint', 'smallint', 'integer', 'bigint'}:
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif data_type in {'real', 'double', 'decimal'}:
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif data_type in {'varchar', 'char'}:
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
    else:
      # TODO(actam): support more types
      # https://prestodb.github.io/docs/current/language/types
      raise RuntimeError(
          'Presto column type {} is not supported.'.format(data_type))
  return tf.train.Example(features=tf.train.Features(feature=feature))


# Create this instead of inline in _PrestoToExample for test mocking purpose.
@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(beam.typehints.Iterable[Tuple[Text, Text,
                                                                Any]])
def _ReadFromPresto(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, client: prestodb.dbapi.Connection,
    query: Text) -> beam.pvalue.PCollection:

  return (pipeline
          | 'Query' >> beam.Create([query])
          | 'QueryTable' >> beam.ParDo(_ReadPrestoDoFn(client)))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _PrestoToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.TfxArtifact]],  # pylint: disable=unused-argument
    exec_properties: Dict[Text, Any],
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read from Presto and transform to TF examples.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, a Presto sql string.

  Returns:
    PCollection of TF examples.
  """
  input_config = example_gen_pb2.Input()
  json_format.Parse(exec_properties['input_config'], input_config)

  client = _deserialize_conn_config(input_config)
  return (pipeline
          | 'QueryTable' >> _ReadFromPresto(client=client, query=split_pattern)  # pylint: disable=no-value-for-parameter
          | 'ToTFExample' >> beam.Map(_row_to_example))


class Executor(base_example_gen_executor.BaseExampleGenExecutor):
  """Generic TFX PrestoExampleGen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for Presto to TF examples."""
    return _PrestoToExample
