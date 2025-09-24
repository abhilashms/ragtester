from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence, Optional

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import (
    retry_with_backoff, validate_model_name, validate_messages_format,
    handle_api_error, handle_rate_limit_response, test_api_connection,
    AuthenticationError, RateLimitError, PermissionError, ServerError,
    ConnectionError, TimeoutError
)


class BedrockLLM(LLMProvider):
    """
    AWS Bedrock LLM provider that supports various Bedrock models including Claude.
    """
    
    def __init__(self, model: str = None, region: str = None, **kwargs: Any) -> None:
        """
        Initialize Bedrock LLM provider.
        
        Args:
            model: Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-20250514-v1:0')
            region: AWS region (defaults to us-east-1)
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.model_id = model or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.region = region or "us-east-1"
        
        # Store chat parameters separately from client parameters
        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024)
        }
        
        # Only add top_p if explicitly provided (Haiku doesn't support it)
        if 'top_p' in kwargs:
            self.chat_params['top_p'] = kwargs['top_p']
        
        # Only pass AWS-specific parameters to client initialization
        self.client_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['temperature', 'max_tokens', 'top_p', 'model', 'region']}
        
        self._client = None
        
        self.logger.debug(f"🚀 Initializing Bedrock LLM Provider")
        self.logger.debug(f"  - Model: {self.model_id}")
        self.logger.debug(f"  - Region: {self.region}")
        self.logger.debug(f"  - Chat params: {self.chat_params}")
        self.logger.debug(f"  - Client kwargs: {self.client_kwargs}")
        
        # Validate and normalize model name
        self.model_id = self._validate_model()
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock client."""
        self.logger.info("🔧 Initializing Bedrock client...")
        self.logger.info(f"  📍 Region: {self.region}")
        self.logger.info(f"  🤖 Model: {self.model_id}")
        
        try:
            self.logger.debug("Importing boto3...")
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            self.logger.debug("✅ boto3 imported successfully")
            
            # Initialize Bedrock client
            self.logger.info(f"Creating Bedrock runtime client for region: {self.region}")
            self.logger.debug(f"Client kwargs: {self.client_kwargs}")
            self._client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                **self.client_kwargs
            )
            self.logger.info("✅ Bedrock runtime client created successfully")
            
            # Test the connection by listing available models
            self.logger.info("🔍 Testing Bedrock connection by listing available models...")
            try:
                bedrock_client = boto3.client('bedrock', region_name=self.region)
                self.logger.debug("Created Bedrock list client")
                
                response = bedrock_client.list_foundation_models()
                available_models = [model['modelId'] for model in response['modelSummaries']]
                self.logger.info(f"✅ Successfully listed {len(available_models)} available models in {self.region}")
                
                # Filter models by provider for better visibility
                anthropic_models = [m for m in available_models if 'anthropic' in m.lower()]
                amazon_models = [m for m in available_models if 'amazon' in m.lower()]
                meta_models = [m for m in available_models if 'meta' in m.lower()]
                
                self.logger.info(f"  🤖 Anthropic models ({len(anthropic_models)}): {anthropic_models}")
                self.logger.info(f"  🏢 Amazon models ({len(amazon_models)}): {amazon_models}")
                self.logger.info(f"  🦙 Meta models ({len(meta_models)}): {meta_models}")
                
                # Check if the model is available (both with and without region prefix)
                model_found = False
                if self.model_id in available_models:
                    model_found = True
                else:
                    # Check if it's available with region prefix
                    for prefix in ['us.', 'eu.', 'ap.']:
                        prefixed_model = prefix + self.model_id
                        if prefixed_model in available_models:
                            model_found = True
                            self.logger.info(f"✅ Model found with region prefix: {prefixed_model}")
                            break
                
                if not model_found:
                    self.logger.warning(f"⚠️ Model {self.model_id} not found in list_foundation_models() response")
                    self.logger.warning(f"This doesn't necessarily mean the model is unavailable - it might:")
                    self.logger.warning(f"  1. Require special permissions or access patterns")
                    self.logger.warning(f"  2. Be available but not listed in foundation models")
                    self.logger.warning(f"  3. Have different access requirements (inference profiles)")
                    self.logger.info(f"Available Anthropic models: {anthropic_models}")
                    self.logger.info(f"💡 Will attempt to use the model anyway - the real test is during invocation")
                else:
                    self.logger.info(f"✅ Model {self.model_id} is listed in available models")
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                self.logger.error(f"❌ Bedrock API error during model listing: {error_code}")
                self.logger.error(f"Error message: {error_message}")
                self.logger.error(f"Full error: {e}")
                
                if error_code == 'AccessDeniedException':
                    self.logger.error("❌ Access denied to Bedrock. Check your IAM permissions.")
                    self.logger.error("Required permissions: bedrock:ListFoundationModels, bedrock:InvokeModel")
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
        
        self.logger.debug("✅ Bedrock client initialization completed successfully")
    
    def _validate_model(self) -> str:
        """Validate and normalize the model name format."""
        
        # Normalize the model ID by removing region prefixes
        normalized_model_id = self.model_id
        if normalized_model_id.startswith('us.'):
            normalized_model_id = normalized_model_id[3:]  # Remove 'us.' prefix
            self.logger.info(f"🔧 Normalized model ID: {self.model_id} -> {normalized_model_id}")
        elif normalized_model_id.startswith('eu.'):
            normalized_model_id = normalized_model_id[3:]  # Remove 'eu.' prefix
            self.logger.info(f"🔧 Normalized model ID: {self.model_id} -> {normalized_model_id}")
        elif normalized_model_id.startswith('ap.'):
            normalized_model_id = normalized_model_id[3:]  # Remove 'ap.' prefix
            self.logger.info(f"🔧 Normalized model ID: {self.model_id} -> {normalized_model_id}")
        
        valid_models = [
            # Anthropic Claude Models
            "anthropic.claude-3-5-sonnet-20241022-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-5-opus-20241022-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-2.1-v1:0",
            "anthropic.claude-2.0-v1:0",
            "anthropic.claude-instant-1.2-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",  # Add the available model
            
            # Amazon Titan Models
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            "amazon.titan-text-agile-v1",
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text-v2",
            "amazon.titan-image-generator-v1",
            "amazon.titan-multimodal-embedding-v1",
            
            # Cohere Models
            "cohere.command-text-v14",
            "cohere.command-light-text-v14",
            "cohere.command-r-plus-v1:0",
            "cohere.command-r-v1:0",
            "cohere.embed-english-v3",
            "cohere.embed-multilingual-v3",
            
            # AI21 Labs Models
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
            "ai21.j2-light-v1",
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
            "ai21.j2-light-v1",
            
            # Meta Llama Models
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "meta.llama3-405b-instruct-v1:0",
            "meta.llama3.1-8b-instruct-v1:0",
            "meta.llama3.1-70b-instruct-v1:0",
            "meta.llama3.1-405b-instruct-v1:0",
            "meta.llama3.1-8b-instruct-v1:0",
            "meta.llama3.1-70b-instruct-v1:0",
            "meta.llama3.1-405b-instruct-v1:0",
            
            # Mistral Models
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral.mixtral-8x22b-instruct-v0:1",
            "mistral.mistral-large-2402-v1:0",
            "mistral.mistral-nemo-12b-instruct-v0:1",
            
            # Stability AI Models
            "stability.stable-diffusion-xl-v1",
            "stability.stable-diffusion-xl-v0",
            "stability.stable-image-remove-background-v1:0",
            "stability.stable-image-style-guide-v1:0",
            "stability.stable-image-control-sketch-v1:0",
            "stability.stable-image-erase-object-v1:0",
            
            # NVIDIA Models
            "nvidia.nemotron-3-8b-instruct-v1-qwq",
            "nvidia.nemotron-3-8b-instruct-v1-qwen",
            
            # Custom Models (if available)
            "custom.model-name-v1:0",
        ]
        
        # Validate the normalized model ID
        validated_model_id = validate_model_name(normalized_model_id, valid_models, "AWS Bedrock")
        
        # Update the instance variable to use the normalized model ID
        self.model_id = validated_model_id
        self.logger.info(f"✅ Using normalized model ID: {validated_model_id}")
        
        return validated_model_id
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
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
        self.logger.info(f"  🤖 Model: {self.model_id}")
        self.logger.info(f"  📍 Region: {self.region}")
        
        # Validate messages format
        validate_messages_format(list(messages), "AWS Bedrock")
        self.logger.debug(f"Messages: {messages}")
        self.logger.debug(f"Chat kwargs: {kwargs}")
        
        if self._client is None:
            self.logger.error("❌ Bedrock client not initialized")
            raise RuntimeError("Bedrock client not initialized")
        
        # Convert messages to the format expected by the model
        self.logger.debug("Formatting messages for Bedrock...")
        formatted_messages = self._format_messages(messages)
        self.logger.debug(f"Formatted messages: {formatted_messages}")
        
        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        self.logger.info(f"📝 Request parameters: {merged_chat_params}")
        
        # Determine the request body format based on model
        if "anthropic" in self.model_id.lower():
            self.logger.info("🔧 Creating Anthropic request format...")
            request_body = self._create_anthropic_request(formatted_messages, merged_chat_params)
        else:
            self.logger.info("🔧 Creating generic request format...")
            # Generic format for other models
            request_body = self._create_generic_request(formatted_messages, merged_chat_params)
        
        self.logger.debug(f"Request body: {request_body}")
        
        try:
            self.logger.info(f"🚀 Invoking Bedrock model: {self.model_id}")
            self.logger.info(f"  📍 Region: {self.region}")
            self.logger.info(f"  📝 Request size: {len(json.dumps(request_body))} bytes")
            
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
            error_msg = str(e)
            self.logger.error(f"❌ Bedrock API call failed: {error_msg}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Model ID: {self.model_id}")
            self.logger.error(f"Region: {self.region}")
            
            # Log detailed error information
            if hasattr(e, 'response'):
                self.logger.error(f"Response status: {e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 'Unknown')}")
                if 'Error' in e.response:
                    error_info = e.response['Error']
                    self.logger.error(f"Error code: {error_info.get('Code', 'Unknown')}")
                    self.logger.error(f"Error message: {error_info.get('Message', 'Unknown')}")
                    if 'RequestId' in e.response.get('ResponseMetadata', {}):
                        self.logger.error(f"Request ID: {e.response['ResponseMetadata']['RequestId']}")
            
            # Provide specific guidance for common errors
            if "on-demand throughput isn't supported" in error_msg:
                self.logger.error("🚨 DETAILED ERROR ANALYSIS:")
                self.logger.error("   ❌ Problem: The model requires inference profiles for on-demand access")
                self.logger.error("   ❌ This model doesn't support direct on-demand invocation")
                self.logger.error("   ❌ AWS Bedrock has different access models for different model variants")
                
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   Option 1: Create an inference profile in AWS Bedrock console")
                self.logger.error("   Option 2: Use a different model that supports on-demand access")
                self.logger.error("   Option 3: Use these models that support on-demand access:")
                self.logger.error("     ✅ anthropic.claude-sonnet-4-20250514-v1:0 (Recommended)")
                self.logger.error("     ✅ anthropic.claude-3-5-sonnet-20241022-v1:0")
                self.logger.error("     ✅ amazon.titan-text-express-v1")
                self.logger.error("     ✅ meta.llama3-8b-instruct-v1:0")
                
                self.logger.error("🔧 IMMEDIATE FIX:")
                self.logger.error("   Option A: Create inference profile for this model in AWS console")
                self.logger.error("   Option B: Change your model configuration to:")
                self.logger.error("   model='anthropic.claude-sonnet-4-20250514-v1:0'")
                
                self.logger.error("📋 INFERENCE PROFILE SETUP:")
                self.logger.error("   1. Go to AWS Bedrock console")
                self.logger.error("   2. Navigate to 'Inference profiles'")
                self.logger.error("   3. Create a new profile with this model")
                self.logger.error("   4. Use the profile ARN instead of model ID")
                
                # Suggest the available model from the error logs
                suggested_model = self._get_suggested_alternative_model()
                if suggested_model:
                    self.logger.error(f"   💡 Recommended: Use '{suggested_model}' (available in your region)")
            
            elif "AccessDeniedException" in error_msg:
                self.logger.error("🚨 ACCESS DENIED ERROR:")
                self.logger.error("   ❌ Your AWS credentials don't have permission to invoke Bedrock models")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Check your IAM permissions")
                self.logger.error("   2. Ensure you have bedrock:InvokeModel permission")
                self.logger.error("   3. Verify your AWS credentials with: aws sts get-caller-identity")
            
            elif "ValidationException" in error_msg:
                self.logger.error("🚨 VALIDATION ERROR:")
                self.logger.error("   ❌ The request parameters are invalid")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Check if the model ID is correct")
                self.logger.error("   2. Verify the model is available in your region")
                self.logger.error("   3. Check request body format")
            
            # Use enhanced error handling
            handle_api_error(e, "AWS Bedrock", f"Model: {self.model_id}, Region: {self.region}")
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> list:
        """Convert messages to format expected by Bedrock."""
        formatted = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                # For Anthropic models, system messages are handled separately
                if isinstance(content, str):
                    formatted.append({"role": "user", "content": f"System: {content}"})
                else:
                    # Handle multimodal system messages
                    formatted.append({"role": "user", "content": f"System: {str(content)}"})
            elif role in ["user", "assistant"]:
                # Handle both text and multimodal content
                formatted.append({"role": role, "content": content})
        
        return formatted
    
    def _create_anthropic_request(self, messages: list, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Create request body for Anthropic models."""
        # Separate system message if present
        system_message = ""
        chat_messages = []
        
        for msg in messages:
            # Handle both text and multimodal content
            content = msg["content"]
            if isinstance(content, str) and msg["role"] == "user" and content.startswith("System: "):
                system_message = content[8:]  # Remove "System: " prefix
            else:
                chat_messages.append(msg)
        
        # Minimal request body for better compatibility
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "messages": chat_messages
        }
        
        # Only add top_p if explicitly provided (some models don't support it)
        if 'top_p' in kwargs:
            request_body["top_p"] = kwargs['top_p']
        
        # Handle system message based on model compatibility
        if system_message:
            # Some models don't support the 'system' field, add as first message instead
            if "haiku" in self.model_id.lower():
                # For Haiku models, add system as first message
                chat_messages.insert(0, {"role": "user", "content": f"System: {system_message}"})
                request_body["messages"] = chat_messages
            else:
                # For other models, use the system field
                request_body["system"] = system_message
        
        return request_body
    
    def _get_suggested_alternative_model(self) -> Optional[str]:
        """Get a suggested alternative model based on the current model and available models."""
        # Based on the error logs, we know this model is available
        if "claude" in self.model_id.lower():
            return "anthropic.claude-sonnet-4-20250514-v1:0"
        elif "titan" in self.model_id.lower():
            return "amazon.titan-text-express-v1"
        elif "llama" in self.model_id.lower():
            return "meta.llama3-8b-instruct-v1:0"
        return None
    
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
