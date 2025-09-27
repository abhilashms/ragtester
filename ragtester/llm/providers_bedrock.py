from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence, Optional
from pathlib import Path
import traceback

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    ClientError = None
    NoCredentialsError = None

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import (
    retry_with_backoff, validate_model_name, validate_messages_format,
    handle_api_error
)


class BedrockLLM(LLMProvider):
    """
    AWS Bedrock LLM provider that supports various Bedrock models including Claude.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "BedrockLLM"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        region = kwargs.get('region', 'default_region')
        instance_key = f"{model}-{region}"

        if instance_key not in cls._instances:
            instance = super(BedrockLLM, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, model: str = None, region: str = None, inference_profile_arn: str = None, **kwargs: Any) -> None:
        """
        Initialize Bedrock LLM provider. This will only run once per unique instance.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        if boto3 is None:
            self.logger.error("❌ boto3 is required for AWS Bedrock but is not installed.")
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")

        self.logger.info("🚀 INITIALIZING BEDROCK LLM PROVIDER (first time only)...")
        self.model_id = model or "anthropic.claude-3-5-haiku-20241022-v1:0"
        self.region = region or "us-east-1"
        self.inference_profile_arn = inference_profile_arn

        if not self.region or not isinstance(self.region, str) or self.region.strip() == "":
            self.logger.warning(f"⚠️ Invalid region '{self.region}', defaulting to 'us-east-1'")
            self.region = "us-east-1"
        
        self.logger.info(f"📤 Resolved parameters: Model ID={self.model_id}, Region={self.region}")

        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024)
        }
        if 'top_p' in kwargs: self.chat_params['top_p'] = kwargs['top_p']

        self.client_kwargs = {k: v for k, v in kwargs.items()
                              if k not in ['temperature', 'max_tokens', 'top_p', 'model', 'region']}

        self._client = None
        self._initialize_client()
        self._initialized = True

    def _initialize_client(self) -> None:
        self.logger.info("🔧 Initializing Bedrock client...")
        try:
            session = self._create_boto3_session()
            
            self.logger.info(f"🔧 CREATING BEDROCK RUNTIME CLIENT in {self.region}")
            self._client = session.client('bedrock-runtime', region_name=self.region, **self.client_kwargs)
            self.logger.info("✅ Bedrock runtime client created successfully")

            self.logger.info("🔍 Testing Bedrock connection by listing available models...")
            bedrock_client = session.client('bedrock', region_name=self.region)
            response = bedrock_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            self.logger.info(f"✅ Successfully listed {len(available_models)} models in {self.region}")

            if self.model_id not in available_models:
                self.logger.warning(f"⚠️ Model {self.model_id} not found in foundation models list. This is often okay; proceeding with invocation attempt.")

        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not found! Ensure the 'default' profile is configured in ~/.aws/credentials.")
            raise RuntimeError("AWS credentials not found. Please configure them.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                raise RuntimeError(f"Access denied to Bedrock in {self.region}. Check IAM permissions for the 'default' profile.")
            else:
                raise RuntimeError(f"AWS error during client initialization: {e}")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Bedrock client initialization: {e}")
            raise

    # --- START FIX: Automatically use 'default' profile ---
    def _create_boto3_session(self) -> 'boto3.Session':
        """Create a boto3 session, defaulting to the 'default' profile if AWS_PROFILE is not set."""
        aws_profile = os.getenv('AWS_PROFILE')
        
        if aws_profile:
            # If an AWS_PROFILE is explicitly set in the environment, honor it.
            self.logger.info(f"  💡 Using explicitly specified AWS profile from environment: '{aws_profile}'")
            return boto3.Session(profile_name=aws_profile)
        else:
            # If no profile is set in the environment, explicitly use the 'default' profile.
            self.logger.info("  💡 AWS_PROFILE environment variable not set. Explicitly using 'default' profile.")
            return boto3.Session(profile_name='default')
    # --- END FIX ---

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Bedrock chat called with {len(messages)} messages")
        validate_messages_format(list(messages), "AWS Bedrock")

        if self._client is None:
            raise RuntimeError("Bedrock client not initialized")

        formatted_messages = self._format_messages(messages)
        merged_chat_params = {**self.chat_params, **kwargs}

        if "anthropic" in self.model_id.lower():
            request_body = self._create_anthropic_request(formatted_messages, merged_chat_params)
        else:
            request_body = self._create_generic_request(formatted_messages, merged_chat_params)

        try:
            model_identifier = self.inference_profile_arn or self.model_id
            self.logger.info(f"🚀 Invoking Bedrock model: {model_identifier}")
            
            response = self._client.invoke_model(
                modelId=model_identifier,
                body=json.dumps(request_body),
                contentType="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))

            if "anthropic" in self.model_id.lower():
                response_text = response_body['content'][0]['text']
            else:
                response_text = response_body.get('generation', '')

            self.logger.info(f"✅ Successfully extracted response from model.")
            return response_text

        except ClientError as e:
            error_message = str(e)
            self.logger.error(f"❌ Bedrock API call failed: {error_message}")
            if "on-demand throughput isn't supported" in error_message:
                self.logger.error("   💡 This model may require a Provisioned Throughput Inference Profile.")
            handle_api_error(e, "AWS Bedrock", f"Model: {self.model_id}")
        except Exception as e:
            self.logger.error(f"❌ An unexpected error occurred in chat(): {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def _format_messages(self, messages: Sequence[LLMMessage]) -> list:
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] in ["user", "assistant"]]

    def _create_anthropic_request(self, messages: list, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "messages": messages,
            **({'top_p': kwargs['top_p']} if 'top_p' in kwargs else {})
        }

    def _create_generic_request(self, messages: list, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\n\nassistant:"
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get('max_tokens', 1024),
                "temperature": kwargs.get('temperature', 0.7)
            }
        }