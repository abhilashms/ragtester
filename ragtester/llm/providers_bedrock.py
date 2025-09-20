from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class BedrockLLM(LLMProvider):
    """
    AWS Bedrock LLM provider that supports various Bedrock models including Claude.
    """
    
    def __init__(self, model: str = None, region: str = None, **kwargs: Any) -> None:
        """
        Initialize Bedrock LLM provider.
        
        Args:
            model: Bedrock model ID (e.g., 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
            region: AWS region (defaults to us-east-1)
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.model_id = model or "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        self.region = region or "us-east-1"
        self.kwargs = kwargs
        self._client = None
        
        self.logger.info(f"🚀 Initializing Bedrock LLM Provider")
        self.logger.info(f"  - Model: {self.model_id}")
        self.logger.info(f"  - Region: {self.region}")
        self.logger.debug(f"  - Additional kwargs: {kwargs}")
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock client."""
        self.logger.info("🔧 Initializing Bedrock client...")
        
        try:
            self.logger.debug("Importing boto3...")
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            self.logger.info("✅ boto3 imported successfully")
            
            # Initialize Bedrock client
            self.logger.info(f"Creating Bedrock runtime client for region: {self.region}")
            self._client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                **self.kwargs
            )
            self.logger.info("✅ Bedrock runtime client created successfully")
            
            # Test the connection by listing available models
            self.logger.info("Testing Bedrock connection by listing available models...")
            try:
                bedrock_client = boto3.client('bedrock', region_name=self.region)
                self.logger.debug("Created Bedrock list client")
                
                response = bedrock_client.list_foundation_models()
                available_models = [model['modelId'] for model in response['modelSummaries']]
                self.logger.info(f"✅ Successfully listed {len(available_models)} available models")
                self.logger.debug(f"Available models: {available_models[:5]}...")  # Show first 5 models
                
                if self.model_id not in available_models:
                    self.logger.warning(f"⚠️ Model {self.model_id} may not be available in region {self.region}")
                    self.logger.warning(f"Available models: {available_models[:5]}...")
                else:
                    self.logger.info(f"✅ Model {self.model_id} is available in region {self.region}")
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                self.logger.error(f"❌ Bedrock API error: {error_code}")
                self.logger.error(f"Error details: {e}")
                
                if error_code == 'AccessDeniedException':
                    self.logger.error("❌ Access denied to Bedrock. Check your IAM permissions.")
                    raise RuntimeError(f"Access denied to Bedrock in region {self.region}. Check your IAM permissions.")
                elif error_code == 'UnauthorizedOperation':
                    self.logger.error("❌ Unauthorized operation. Check your AWS credentials and permissions.")
                    raise RuntimeError(f"Unauthorized operation. Check your AWS credentials and permissions.")
                else:
                    self.logger.warning(f"⚠️ Could not verify model availability: {e}")
            except NoCredentialsError:
                self.logger.error("❌ AWS credentials not found!")
                self.logger.error("Please configure AWS credentials using:")
                self.logger.error("  - AWS CLI: aws configure")
                self.logger.error("  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
                self.logger.error("  - IAM roles (if running on EC2)")
                raise RuntimeError("AWS credentials not found. Please configure AWS credentials.")
                
        except ImportError as e:
            self.logger.error(f"❌ Import error: {e}")
            self.logger.error("boto3 is required for AWS Bedrock.")
            self.logger.error("Install with: pip install boto3")
            raise ImportError(
                "boto3 is required for AWS Bedrock. "
                "Install with: pip install boto3"
            )
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Bedrock client initialization: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise
        
        self.logger.info("✅ Bedrock client initialization completed successfully")
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Bedrock model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        self.logger.info(f"💬 Bedrock chat called with {len(messages)} messages")
        self.logger.debug(f"Messages: {messages}")
        self.logger.debug(f"Chat kwargs: {kwargs}")
        
        if self._client is None:
            self.logger.error("❌ Bedrock client not initialized")
            raise RuntimeError("Bedrock client not initialized")
        
        # Convert messages to the format expected by the model
        self.logger.debug("Formatting messages for Bedrock...")
        formatted_messages = self._format_messages(messages)
        self.logger.debug(f"Formatted messages: {formatted_messages}")
        
        # Determine the request body format based on model
        if "anthropic" in self.model_id.lower():
            self.logger.debug("Creating Anthropic request format...")
            request_body = self._create_anthropic_request(formatted_messages, kwargs)
        else:
            self.logger.debug("Creating generic request format...")
            # Generic format for other models
            request_body = self._create_generic_request(formatted_messages, kwargs)
        
        self.logger.debug(f"Request body: {request_body}")
        
        try:
            self.logger.info(f"🚀 Invoking Bedrock model: {self.model_id}")
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            self.logger.info("✅ Bedrock API call successful")
            
            response_body = json.loads(response['body'].read())
            self.logger.debug(f"Response body: {response_body}")
            
            # Extract text based on model type
            if "anthropic" in self.model_id.lower():
                response_text = response_body['content'][0]['text']
                self.logger.info(f"✅ Extracted Anthropic response: {response_text[:100]}...")
            elif "amazon" in self.model_id.lower():
                response_text = response_body['generation']
                self.logger.info(f"✅ Extracted Amazon response: {response_text[:100]}...")
            else:
                # Generic extraction
                response_text = response_body.get('outputs', [{}])[0].get('text', '')
                self.logger.info(f"✅ Extracted generic response: {response_text[:100]}...")
            
            return response_text
                
        except Exception as e:
            self.logger.error(f"❌ Bedrock API call failed: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Model ID: {self.model_id}")
            self.logger.error(f"Region: {self.region}")
            raise RuntimeError(f"Bedrock API call failed: {e}")
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> list:
        """Convert messages to format expected by Bedrock."""
        formatted = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                # For Anthropic models, system messages are handled separately
                formatted.append({"role": "user", "content": f"System: {content}"})
            elif role in ["user", "assistant"]:
                formatted.append({"role": role, "content": content})
        
        return formatted
    
    def _create_anthropic_request(self, messages: list, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Create request body for Anthropic models."""
        # Separate system message if present
        system_message = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "user" and msg["content"].startswith("System: "):
                system_message = msg["content"][8:]  # Remove "System: " prefix
            else:
                chat_messages.append(msg)
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0),
            "messages": chat_messages
        }
        
        if system_message:
            request_body["system"] = system_message
        
        return request_body
    
    def _create_generic_request(self, messages: list, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Create request body for non-Anthropic models."""
        # Convert messages to text format
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(text_parts) + "\n\nAssistant:"
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get('max_tokens', 1024),
                "temperature": kwargs.get('temperature', 0.7),
                "topP": kwargs.get('top_p', 0.9),
                "stopSequences": kwargs.get('stop_sequences', [])
            }
        }
