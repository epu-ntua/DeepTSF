import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence
import boto3
from botocore.errorfactory import ClientError
from dagster_aws.s3.compute_log_manager import S3ComputeLogManager
from dagster import (
    Field,
    Permissive,
    StringSource,
)
from dagster._config.config_type import Noneable
from dagster._core.storage.compute_log_manager import CapturedLogContext, ComputeIOType
from dagster._serdes import ConfigurableClassData
from typing_extensions import Self

class MyS3ComputeLogManager(S3ComputeLogManager):
    """Custom class for S3ComputeLogManager with added support for accessKey and secretKey"""

    def __init__(
        self,
        bucket,
        local_dir=None,
        inst_data: Optional[ConfigurableClassData] = None,
        prefix="dagster",
        use_ssl=True,
        verify=True,
        verify_cert_path=None,
        endpoint_url=None,
        skip_empty_files=False,
        upload_interval=None,
        upload_extra_args=None,
        show_url_only=False,
        region=None,
        access_key=None, 
        secret_key=None,  
    ):
        super().__init__(
            bucket=bucket,
            local_dir=local_dir,
            inst_data=inst_data,
            prefix=prefix,
            use_ssl=use_ssl,
            verify=verify,
            verify_cert_path=verify_cert_path,
            endpoint_url=endpoint_url,
            skip_empty_files=skip_empty_files,
            upload_interval=upload_interval,
            upload_extra_args=upload_extra_args,
            show_url_only=show_url_only,
            region=region,
        )

        verify_param = verify_cert_path if verify_cert_path else verify

        if access_key and secret_key:
            self._s3_session = boto3.resource(
                "s3",
                use_ssl=use_ssl,
                verify=verify_param,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key, 
            ).meta.client
        else:
            self._s3_session = boto3.resource(
                "s3", use_ssl=use_ssl, verify=verify_param, endpoint_url=endpoint_url
            ).meta.client

    @classmethod
    def config_type(cls):
        return {
            "bucket": StringSource,
            "local_dir": Field(StringSource, is_required=False),
            "prefix": Field(StringSource, is_required=False, default_value="dagster"),
            "use_ssl": Field(bool, is_required=False, default_value=True),
            "verify": Field(bool, is_required=False, default_value=True),
            "verify_cert_path": Field(StringSource, is_required=False),
            "endpoint_url": Field(StringSource, is_required=False),
            "skip_empty_files": Field(bool, is_required=False, default_value=False),
            "upload_interval": Field(Noneable(int), is_required=False, default_value=None),
            "upload_extra_args": Field(
                Permissive(), is_required=False, description="Extra args for S3 file upload"
            ),
            "show_url_only": Field(bool, is_required=False, default_value=False),
            "region": Field(StringSource, is_required=False),
            "access_key": Field(StringSource, is_required=False),  
            "secret_key": Field(StringSource, is_required=False), 
        }

    @classmethod
    def from_config_value(
        cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]
    ) -> Self:
        return cls(inst_data=inst_data, **config_value)