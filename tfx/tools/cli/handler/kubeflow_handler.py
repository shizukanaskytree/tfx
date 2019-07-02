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
"""Handler for Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import subprocess
import sys
import tempfile

import click
import kfp
import tensorflow as tf
from typing import Text, Dict, Any

from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  # TODO(b/132286477): Update comments after updating methods.

  def __init__(self, flags_dict):
    self.flags_dict = flags_dict
    self._handler_home_dir = self._get_handler_home()

    # Create client
    self._client = kfp.Client(
        host=self.flags_dict[labels.ENDPOINT],
        client_id=self.flags_dict[labels.IAP_CLIENT_ID],
        namespace=self.flags_dict[labels.NAMESPACE])

  def create_pipeline(self, overwrite=False) -> None:
    """Creates pipeline in Kubeflow."""
    self._check_pipeline_dsl_path()
    pipeline_args = self._extract_pipeline_args()
    self._check_pipeline_package_path()

    # Path to pipeline folder in kubeflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    if overwrite:
      # For update, check if pipeline exists.
      if not tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} does not exist.'.format(
            pipeline_args[labels.PIPELINE_NAME]))
    else:
      # For create, verify that pipeline does not exist.
      if tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} already exists.'.format(
            pipeline_args[labels.PIPELINE_NAME]))

    self._save_pipeline(pipeline_args)

  def update_pipeline(self) -> None:
    """Updates pipeline in Kubeflow."""
    # Set overwrite to true for update to make sure pipeline exists.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    click.echo('List of pipelines in Kubeflow')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""
    click.echo('Deleting pipeline in Kubeflow')

  def run_pipeline(self) -> None:
    """Run pipeline in Kubeflow."""
    click.echo('Triggering pipeline in Kubeflow')

  def _get_handler_home(self) -> Text:
    """Sets handler home.

    Returns:
      Path to handler home directory.
    """
    handler_home_dir = 'KUBEFLOW_HOME'
    if handler_home_dir in os.environ:
      return os.environ[handler_home_dir]
    return os.path.join(os.environ['HOME'], 'kubeflow_pipelines', '')

  def _get_handler_pipeline_path(self, pipeline_name) -> Text:
    """Path to pipeline folder in Kubeflow.

    Args:
      pipeline_name: name of the pipeline

    Returns:
      Path to pipeline folder in Kubeflow.
    """
    # Path to pipeline folder in Kubeflow.
    return os.path.join(self._handler_home_dir, pipeline_name, '')

  def _extract_pipeline_args(self) -> Dict[Text, Any]:
    """Get pipeline args from the DSL."""
    if os.path.isdir(self.flags_dict[labels.PIPELINE_DSL_PATH]):
      sys.exit('Provide dsl file path.')

    # Create an environment for subprocess.
    temp_env = os.environ.copy()

    # Create temp file to store pipeline_args from pipeline dsl.
    temp_file = tempfile.mkstemp(prefix='cli_tmp_', suffix='_pipeline_args')[1]

    # Store temp_file path in temp_env.
    temp_env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH] = temp_file

    # Run dsl with mock environment to store pipeline args in temp_file.
    subprocess.call(['python', self.flags_dict[labels.PIPELINE_DSL_PATH]],
                    env=temp_env)

    if os.stat(temp_file).st_size != 0:
      # Load pipeline_args from temp_file for TFX pipelines
      with open(temp_file, 'r') as f:
        pipeline_args = json.load(f)
    else:
      # For non-TFX pipelines, extract pipeline name from the dsl filename.
      pipeline_args = {
          labels.PIPELINE_NAME:
              os.path.basename(self.flags_dict[labels.PIPELINE_DSL_PATH]
                              ).split('.')[0]
      }

    # Delete temp file
    io_utils.delete_dir(temp_file)

    return pipeline_args

  def _save_pipeline(self, pipeline_args) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    # For non-TFX pipelines create an empty pipeline_args.json
    # Path to pipeline folder in Kubeflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    # If updating pipeline, first delete pipeline directory.
    if tf.io.gfile.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Upload pipeline.
    upload_response = self._client.upload_pipeline(
        pipeline_package_path=self.flags_dict[labels.PIPELINE_PACKAGE_PATH],
        pipeline_name=pipeline_args[labels.PIPELINE_NAME])
    click.echo(upload_response)

    # Copy pipeline_args to pipeline folder.
    pipeline_args['kubeflow_pipeline_details'] = upload_response.__dict__
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(os.path.join(handler_pipeline_path, 'pipeline_args.json'),
              'w') as f:
      json.dump(pipeline_args, f)

  def _check_pipeline_package_path(self):
    if not self.flags_dict[labels.PIPELINE_PACKAGE_PATH]:
      sys.exit('Provide the output workflow package path.')

    if not tf.io.gfile.exists(self.flags_dict[labels.PIPELINE_PACKAGE_PATH]):
      sys.exit('Invalid pipeline package path: {}'.format(
          self.flags_dict[labels.PIPELINE_PACKAGE_PATH]))
