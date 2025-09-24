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
    
    def __init__(self, model: str = None, region: str = None, inference_profile_arn: str = None, auto_inference_profile: bool = True, **kwargs: Any) -> None:
        """
        Initialize Bedrock LLM provider.
        
        Args:
            model: Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-20250514-v1:0')
            region: AWS region (defaults to us-east-1)
            inference_profile_arn: Inference profile ARN (if model requires inference profile)
            auto_inference_profile: Whether to automatically find and use inference profiles (default: True)
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.logger.info("🚀 INITIALIZING BEDROCK LLM PROVIDER")
        self.logger.info("=" * 50)
        self.logger.info(f"📥 Input parameters:")
        self.logger.info(f"  🤖 Model: {model}")
        self.logger.info(f"  📍 Region: {region}")
        self.logger.info(f"  🔗 Inference Profile ARN: {inference_profile_arn}")
        self.logger.info(f"  ⚙️ Additional kwargs: {kwargs}")
        
        self.model_id = model or "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        self.region = region or "us-east-1"
        self.inference_profile_arn = inference_profile_arn
        self.auto_inference_profile = auto_inference_profile
        
        # Validate region format
        if not self.region or not isinstance(self.region, str) or self.region.strip() == "":
            original_region = self.region
            self.logger.warning(f"⚠️ Invalid region '{original_region}', defaulting to 'us-east-1'")
            self.region = "us-east-1"
        
        # Common AWS regions for Bedrock
        valid_regions = [
            "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "ap-northeast-1",
            "ca-central-1", "eu-central-1", "eu-west-3", "ap-southeast-2", "ap-south-1"
        ]
        
        if self.region not in valid_regions:
            self.logger.warning(f"⚠️ Region '{self.region}' may not support Bedrock. Common regions: {valid_regions[:5]}")
            self.logger.warning(f"   Using '{self.region}' anyway - availability varies by model")
        
        self.logger.info(f"📤 Resolved parameters:")
        self.logger.info(f"  🤖 Model ID: {self.model_id}")
        self.logger.info(f"  📍 Region: {self.region}")
        if self.inference_profile_arn:
            self.logger.info(f"  🔗 Inference Profile ARN: {self.inference_profile_arn}")
            self.logger.info(f"  ✅ Using inference profile for model access")
        
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
        
        # Initialize Bedrock client (unless explicitly skipped for testing)
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
            
            # Check AWS credentials
            self.logger.info("🔑 CHECKING AWS CREDENTIALS")
            self.logger.info("-" * 30)
            try:
                # Try to get AWS session info
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials:
                    self.logger.info("✅ AWS credentials found")
                    self.logger.info(f"  🔑 Access Key ID: {credentials.access_key[:4]}...{credentials.access_key[-4:]}")
                    self.logger.info(f"  🔑 Secret Key: {'*' * 20}")
                    self.logger.info(f"  🔑 Session Token: {'Present' if credentials.token else 'None'}")
                else:
                    self.logger.warning("⚠️ No AWS credentials found in session")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not check credentials: {e}")
            
            # Initialize Bedrock client
            self.logger.info(f"🔧 CREATING BEDROCK RUNTIME CLIENT")
            self.logger.info("-" * 30)
            self.logger.info(f"  📍 Region: {self.region}")
            self.logger.info(f"  ⚙️ Client kwargs: {self.client_kwargs}")
            self.logger.info(f"  🔧 Service: bedrock-runtime")
            
            # Handle AWS profile configuration
            aws_profile = os.getenv('AWS_PROFILE', 'default')
            self.logger.info(f"  🔑 AWS Profile: {aws_profile}")
            
            if aws_profile == 'default' and not os.getenv('AWS_PROFILE'):
                self.logger.info(f"  💡 Using default AWS profile (no AWS_PROFILE env var set)")
                self.logger.info(f"  💡 To use a specific profile, set: export AWS_PROFILE=your-profile-name")
            
            # Create boto3 session with explicit profile
            session = boto3.Session(profile_name=aws_profile)
            self._client = session.client(
                'bedrock-runtime',
                region_name=self.region,
                **self.client_kwargs
            )
            self.logger.info("✅ Bedrock runtime client created successfully")
            self.logger.info(f"  🆔 Client endpoint: {self._client._endpoint}")
            self.logger.info(f"  📍 Client region: {self._client._client_config.region_name}")
            
            # Test the connection by listing available models
            self.logger.info("🔍 Testing Bedrock connection by listing available models...")
            try:
                bedrock_client = session.client('bedrock', region_name=self.region)
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
                self.logger.error("  - AWS Profile: export AWS_PROFILE=your-profile-name")
                self.logger.error("")
                self.logger.error("💡 TIP: If you have multiple AWS profiles, try:")
                self.logger.error("   export AWS_PROFILE=default")
                self.logger.error("   or")
                self.logger.error("   export AWS_PROFILE=your-specific-profile")
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
        
        # Store original model ID for logging
        original_model_id = self.model_id
        self.logger.info(f"🔧 Original model ID: {original_model_id}")
        
        # Handle the :0 suffix issue - some models with :0 suffix require inference profiles
        normalized_model_id = original_model_id
        
        # Check if this is a model that requires inference profiles
        if self._requires_inference_profile(normalized_model_id):
            if self.inference_profile_arn:
                # User has provided an inference profile ARN
                self.logger.info(f"✅ Using provided inference profile for model {normalized_model_id}")
                self.logger.info(f"   🔗 Inference Profile ARN: {self.inference_profile_arn}")
                self.logger.info(f"   ✅ This should work with your configured inference profile")
                # Don't change the model ID - use the original with the inference profile
            elif self.auto_inference_profile:
                # Try to automatically find and use an inference profile
                self.logger.info(f"🔍 Auto-detecting inference profile for model {normalized_model_id}")
                auto_profile_arn = self._auto_discover_inference_profile(normalized_model_id)
                
                if auto_profile_arn:
                    self.logger.info(f"✅ Auto-discovered inference profile!")
                    self.logger.info(f"   🔗 Inference Profile ARN: {auto_profile_arn}")
                    self.logger.info(f"   🚀 Using inference profile automatically")
                    self.inference_profile_arn = auto_profile_arn
                else:
                    # No inference profile found, suggest alternatives
                    self.logger.warning(f"⚠️ No inference profile found for {normalized_model_id}")
                    self.logger.warning(f"   This model doesn't support on-demand throughput")
                    
                    # Try to suggest an alternative without :0 suffix
                    alternative_model = self._get_alternative_model(normalized_model_id)
                    if alternative_model:
                        self.logger.info(f"💡 SUGGESTED ALTERNATIVE: {alternative_model}")
                        self.logger.info(f"   This model supports on-demand throughput")
                        self.logger.info(f"   You can use this model instead or create an inference profile")
                        
                        # Check if the alternative is actually available
                        if self._check_model_availability(alternative_model):
                            self.logger.info(f"🔄 Auto-switching to alternative model: {alternative_model}")
                            normalized_model_id = alternative_model
                        else:
                            self.logger.warning(f"⚠️ Suggested alternative {alternative_model} is not available")
                            self.logger.warning(f"   Will continue with original model - may require inference profile")
                    else:
                        self.logger.error(f"❌ No alternative model found for {normalized_model_id}")
                        self.logger.error(f"   Please create an inference profile in AWS Bedrock console")
                        self.logger.error(f"   Or use a different model that supports on-demand access")
                        self.logger.error(f"   Available models vary by region and account permissions")
                        
                        # List some available models to help user
                        self._log_available_models_suggestion()
            else:
                # Auto-discovery disabled, suggest alternatives
                self.logger.warning(f"⚠️ Model {normalized_model_id} may require an inference profile")
                self.logger.warning(f"   This model doesn't support on-demand throughput")
                self.logger.warning(f"   Auto-discovery is disabled. Enable with auto_inference_profile=True")
                
                # Try to suggest an alternative without :0 suffix
                alternative_model = self._get_alternative_model(normalized_model_id)
                if alternative_model:
                    self.logger.info(f"💡 SUGGESTED ALTERNATIVE: {alternative_model}")
                    self.logger.info(f"   This model supports on-demand throughput")
                    self.logger.info(f"   You can use this model instead or create an inference profile")
                    
                    # Check if the alternative is actually available
                    if self._check_model_availability(alternative_model):
                        self.logger.info(f"🔄 Auto-switching to alternative model: {alternative_model}")
                        normalized_model_id = alternative_model
                    else:
                        self.logger.warning(f"⚠️ Suggested alternative {alternative_model} is not available")
                        self.logger.warning(f"   Will continue with original model - may require inference profile")
                else:
                    self.logger.error(f"❌ No alternative model found for {normalized_model_id}")
                    self.logger.error(f"   Please create an inference profile in AWS Bedrock console")
                    self.logger.error(f"   Or use a different model that supports on-demand access")
                    self.logger.error(f"   Available models vary by region and account permissions")
                    
                    # List some available models to help user
                    self._log_available_models_suggestion()
        
        valid_models = [
            # Anthropic Claude Models (without region prefix)
            "anthropic.claude-3-5-sonnet-20241022-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",  # Requires inference profile
            "anthropic.claude-3-5-opus-20241022-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-2.1-v1:0",
            "anthropic.claude-2.0-v1:0",
            "anthropic.claude-instant-1.2-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",  # Supports on-demand access
            
            # Anthropic Claude Models (with US region prefix) - These keep their prefix
            "us.anthropic.claude-3-5-sonnet-20241022-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",  # User's model - requires inference profile
            "us.anthropic.claude-3-5-opus-20241022-v1:0",
            "us.anthropic.claude-3-opus-20240229-v1:0",
            "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "us.anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-2.1-v1:0",
            "us.anthropic.claude-2.0-v1:0",
            "us.anthropic.claude-instant-1.2-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",  # Alternative that supports on-demand access
            
            # Anthropic Claude Models (with EU region prefix)
            "eu.anthropic.claude-3-5-sonnet-20241022-v1:0",
            "eu.anthropic.claude-3-5-haiku-20241022-v1:0",  # Requires inference profile
            "eu.anthropic.claude-3-5-opus-20241022-v1:0",
            "eu.anthropic.claude-3-opus-20240229-v1:0",
            "eu.anthropic.claude-3-sonnet-20240229-v1:0",
            "eu.anthropic.claude-3-haiku-20240307-v1:0",
            "eu.anthropic.claude-2.1-v1:0",
            "eu.anthropic.claude-2.0-v1:0",
            "eu.anthropic.claude-instant-1.2-v1:0",
            "eu.anthropic.claude-sonnet-4-20250514-v1:0",  # Alternative that supports on-demand access
            
            # Anthropic Claude Models (with AP region prefix)
            "ap.anthropic.claude-3-5-sonnet-20241022-v1:0",
            "ap.anthropic.claude-3-5-haiku-20241022-v1:0",  # Requires inference profile
            "ap.anthropic.claude-3-5-opus-20241022-v1:0",
            "ap.anthropic.claude-3-opus-20240229-v1:0",
            "ap.anthropic.claude-3-sonnet-20240229-v1:0",
            "ap.anthropic.claude-3-haiku-20240307-v1:0",
            "ap.anthropic.claude-2.1-v1:0",
            "ap.anthropic.claude-2.0-v1:0",
            "ap.anthropic.claude-instant-1.2-v1:0",
            "ap.anthropic.claude-sonnet-4-20250514-v1:0",  # Alternative that supports on-demand access
            
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
            # Determine what to use as the model identifier
            model_identifier = self.inference_profile_arn if self.inference_profile_arn else self.model_id
            
            self.logger.info(f"🚀 Invoking Bedrock model: {model_identifier}")
            self.logger.info(f"  📍 Region: {self.region}")
            self.logger.info(f"  📝 Request size: {len(json.dumps(request_body))} bytes")
            self.logger.info(f"  🔧 Content type: application/json")
            if self.inference_profile_arn:
                self.logger.info(f"  🔗 Using inference profile ARN")
            else:
                self.logger.info(f"  🤖 Using direct model ID")
            
            # Log the exact request being sent
            self.logger.debug(f"🔍 EXACT REQUEST BEING SENT:")
            self.logger.debug(f"  modelId: {model_identifier}")
            self.logger.debug(f"  body: {json.dumps(request_body, indent=2)}")
            self.logger.debug(f"  contentType: application/json")
            
            response = self._client.invoke_model(
                modelId=model_identifier,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            
            # Log response metadata
            self.logger.info("✅ Bedrock API call successful")
            self.logger.info(f"  📊 Response status: {response.get('ResponseMetadata', {}).get('HTTPStatusCode', 'Unknown')}")
            self.logger.info(f"  🆔 Request ID: {response.get('ResponseMetadata', {}).get('RequestId', 'Unknown')}")
            self.logger.info(f"  📏 Response size: {len(response.get('body', b''))} bytes")
            
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
            
            # Log comprehensive error information
            self.logger.error(f"🔍 DETAILED ERROR ANALYSIS:")
            self.logger.error(f"  📍 Full error message: {error_msg}")
            self.logger.error(f"  🏷️ Error type: {type(e).__name__}")
            self.logger.error(f"  📍 Module: {e.__class__.__module__}")
            self.logger.error(f"  📍 Class: {e.__class__.__name__}")
            
            # Log request details that failed
            self.logger.error(f"🔍 REQUEST DETAILS THAT FAILED:")
            self.logger.error(f"  🤖 Model ID: {self.model_id}")
            self.logger.error(f"  📍 Region: {self.region}")
            self.logger.error(f"  📝 Request body: {json.dumps(request_body, indent=2)}")
            self.logger.error(f"  🔧 Content type: application/json")
            
            # Log detailed error information
            if hasattr(e, 'response'):
                self.logger.error(f"🔍 AWS RESPONSE DETAILS:")
                self.logger.error(f"  📊 Response status: {e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 'Unknown')}")
                self.logger.error(f"  🆔 Request ID: {e.response.get('ResponseMetadata', {}).get('RequestId', 'Unknown')}")
                self.logger.error(f"  📏 Response size: {len(str(e.response))} bytes")
                
                if 'Error' in e.response:
                    error_info = e.response['Error']
                    self.logger.error(f"  🚨 Error code: {error_info.get('Code', 'Unknown')}")
                    self.logger.error(f"  📝 Error message: {error_info.get('Message', 'Unknown')}")
                    self.logger.error(f"  🏷️ Error type: {error_info.get('Type', 'Unknown')}")
                    
                    # Log additional error details if available
                    if 'Detail' in error_info:
                        self.logger.error(f"  📋 Error details: {error_info['Detail']}")
                
                # Log full response for debugging
                self.logger.debug(f"🔍 FULL AWS RESPONSE:")
                self.logger.debug(f"  {json.dumps(e.response, indent=2, default=str)}")
            
            # Log stack trace for debugging
            import traceback
            self.logger.debug(f"🔍 FULL STACK TRACE:")
            self.logger.debug(f"  {traceback.format_exc()}")
            
            # Provide specific guidance for common errors
            if "on-demand throughput isn't supported" in error_msg:
                self.logger.error("🚨 DETAILED ERROR ANALYSIS:")
                self.logger.error("   ❌ Problem: The model requires inference profiles for on-demand access")
                self.logger.error("   ❌ This model doesn't support direct on-demand invocation")
                self.logger.error("   ❌ AWS Bedrock has different access models for different model variants")
                
                # Get the current model and suggest alternatives
                current_model = self.model_id
                alternative_model = self._get_alternative_model(current_model)
                
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   Option 1: Create an inference profile in AWS Bedrock console")
                self.logger.error("   Option 2: Use a different model that supports on-demand access")
                self.logger.error("   Option 3: Use these models that support on-demand access:")
                self.logger.error("     ✅ anthropic.claude-sonnet-4-20250514-v1:0 (Recommended)")
                self.logger.error("     ✅ anthropic.claude-3-5-sonnet-20241022-v1:0")
                self.logger.error("     ✅ amazon.titan-text-express-v1")
                self.logger.error("     ✅ meta.llama3-8b-instruct-v1:0")
                
                if alternative_model:
                    self.logger.error(f"🔧 IMMEDIATE FIX - AUTO-SUGGESTED ALTERNATIVE:")
                    self.logger.error(f"   💡 Use '{alternative_model}' instead of '{current_model}'")
                    self.logger.error(f"   ✅ This model supports on-demand access and preserves your region prefix")
                    self.logger.error(f"   🔄 The system will automatically switch to this model on next run")
                else:
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
                self.logger.error("   4. Ensure the model supports on-demand access")
                
                # Check for specific validation issues
                if "model" in error_msg.lower() and "not found" in error_msg.lower():
                    self.logger.error("🔍 MODEL NOT FOUND ISSUE:")
                    self.logger.error("   ❌ The specified model is not available in your region")
                    self.logger.error("   💡 Try using a different region or model")
                    self.logger.error("   💡 Available models vary by region")
                
                elif "throughput" in error_msg.lower():
                    self.logger.error("🔍 THROUGHPUT ISSUE:")
                    self.logger.error("   ❌ Model throughput configuration issue")
                    self.logger.error("   💡 This model may require inference profiles")
                    self.logger.error("   💡 Or use a different model that supports on-demand access")
            
            elif "ThrottlingException" in error_msg or "throttling" in error_msg.lower():
                self.logger.error("🚨 THROTTLING ERROR:")
                self.logger.error("   ❌ Request rate exceeded for Bedrock")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Wait before retrying the request")
                self.logger.error("   2. Reduce request frequency")
                self.logger.error("   3. Consider using provisioned throughput for high-volume usage")
            
            elif "ModelNotReadyException" in error_msg:
                self.logger.error("🚨 MODEL NOT READY ERROR:")
                self.logger.error("   ❌ The model is currently not available")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Wait a few minutes and try again")
                self.logger.error("   2. Check AWS service status")
                self.logger.error("   3. Try a different model")
            
            elif "ModelTimeoutException" in error_msg:
                self.logger.error("🚨 MODEL TIMEOUT ERROR:")
                self.logger.error("   ❌ The model request timed out")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Reduce the input size")
                self.logger.error("   2. Increase timeout settings")
                self.logger.error("   3. Try a different model with faster response times")
            
            elif "ServiceQuotaExceededException" in error_msg:
                self.logger.error("🚨 SERVICE QUOTA EXCEEDED:")
                self.logger.error("   ❌ You've exceeded your service quota")
                self.logger.error("💡 SOLUTIONS:")
                self.logger.error("   1. Request a quota increase in AWS console")
                self.logger.error("   2. Wait for quota reset period")
                self.logger.error("   3. Use different models or regions")
            
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
    
    def _requires_inference_profile(self, model_id: str) -> bool:
        """Check if a model requires an inference profile for on-demand access."""
        # Models that are known to require inference profiles
        inference_profile_required = [
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "eu.anthropic.claude-3-5-haiku-20241022-v1:0",
            "ap.anthropic.claude-3-5-haiku-20241022-v1:0",
        ]
        return model_id in inference_profile_required
    
    def _get_alternative_model(self, model_id: str) -> Optional[str]:
        """Get an alternative model that supports on-demand access."""
        # Try to find an available alternative from the models we know are available
        available_alternatives = self._get_available_alternative_models()
        
        if not available_alternatives:
            # Fallback to hardcoded alternatives if we can't determine availability
            alternatives = {
                "anthropic.claude-3-5-haiku-20241022-v1:0": "anthropic.claude-3-5-sonnet-20241022-v1:0",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0": "us.anthropic.claude-3-5-sonnet-20241022-v1:0",
                "eu.anthropic.claude-3-5-haiku-20241022-v1:0": "eu.anthropic.claude-3-5-sonnet-20241022-v1:0",
                "ap.anthropic.claude-3-5-haiku-20241022-v1:0": "ap.anthropic.claude-3-5-sonnet-20241022-v1:0",
            }
            return alternatives.get(model_id)
        
        # Find the best alternative from available models
        return self._find_best_alternative(model_id, available_alternatives)
    
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
    
    def _get_available_alternative_models(self) -> list:
        """Get list of available alternative models that support on-demand access."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create a temporary client for checking model availability
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
            temp_client = session.client('bedrock', region_name=self.region)
            
            # List foundation models to check availability
            response = temp_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            
            # Filter for models that typically support on-demand access
            on_demand_models = [
                "anthropic.claude-3-5-sonnet-20241022-v1:0",
                "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-sonnet-4-20250514-v1:0",
                "amazon.titan-text-express-v1",
                "amazon.titan-text-lite-v1",
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-70b-instruct-v1:0",
                "cohere.command-text-v14",
                "cohere.command-light-text-v14",
            ]
            
            # Add region-prefixed versions
            region_prefixes = ['us.', 'eu.', 'ap.']
            for prefix in region_prefixes:
                on_demand_models.extend([prefix + model for model in on_demand_models if not model.startswith(prefix)])
            
            # Filter to only include available models
            available_alternatives = [model for model in on_demand_models if model in available_models]
            
            if available_alternatives:
                self.logger.info(f"✅ Found {len(available_alternatives)} available alternative models")
            else:
                self.logger.warning(f"⚠️ No alternative models found in available models")
            
            return available_alternatives
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get available alternative models: {e}")
            return []
    
    def _find_best_alternative(self, original_model: str, available_alternatives: list) -> Optional[str]:
        """Find the best alternative model based on the original model."""
        if not available_alternatives:
            return None
        
        # Extract region prefix from original model
        region_prefix = ""
        for prefix in ['us.', 'eu.', 'ap.']:
            if original_model.startswith(prefix):
                region_prefix = prefix
                break
        
        # Prefer models with the same region prefix
        if region_prefix:
            prefixed_alternatives = [model for model in available_alternatives if model.startswith(region_prefix)]
            if prefixed_alternatives:
                # Prefer Claude models if original is Claude
                if 'claude' in original_model.lower():
                    claude_alternatives = [model for model in prefixed_alternatives if 'claude' in model.lower()]
                    if claude_alternatives:
                        return claude_alternatives[0]
                return prefixed_alternatives[0]
        
        # Fallback to any available alternative
        # Prefer Claude models if original is Claude
        if 'claude' in original_model.lower():
            claude_alternatives = [model for model in available_alternatives if 'claude' in model.lower()]
            if claude_alternatives:
                return claude_alternatives[0]
        
        return available_alternatives[0]
    
    def _auto_discover_inference_profile(self, model_id: str) -> Optional[str]:
        """
        Automatically discover and return an inference profile ARN for the given model.
        
        Args:
            model_id: The model ID that requires an inference profile
            
        Returns:
            The inference profile ARN if found, None otherwise
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.logger.info(f"🔍 Searching for inference profiles in region {self.region}")
            
            # Create a temporary client for checking inference profiles
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
            temp_client = session.client('bedrock', region_name=self.region)
            
            # List inference profiles
            response = temp_client.list_inference_profiles()
            profiles = response.get('inferenceProfileSummaries', [])
            
            if not profiles:
                self.logger.warning(f"⚠️ No inference profiles found in region {self.region}")
                return None
            
            self.logger.info(f"✅ Found {len(profiles)} inference profile(s)")
            
            # Look for profiles that contain the target model
            matching_profiles = []
            
            for profile in profiles:
                profile_arn = profile.get('inferenceProfileArn', '')
                profile_name = profile.get('inferenceProfileName', '')
                status = profile.get('status', '')
                
                if status != 'ACTIVE':
                    continue
                
                try:
                    # Get detailed information about the profile
                    detail_response = temp_client.get_inference_profile(inferenceProfileIdentifier=profile_arn)
                    detail = detail_response.get('inferenceProfile', {})
                    foundation_models = detail.get('foundationModelArns', [])
                    
                    # Check if this profile contains the target model
                    for model_arn in foundation_models:
                        # Extract model ID from ARN
                        model_id_from_arn = model_arn.split('/')[-1] if '/' in model_arn else model_arn
                        
                        # Check for exact match or similar model
                        if (model_id_from_arn == model_id or 
                            self._is_model_compatible(model_id, model_id_from_arn)):
                            matching_profiles.append({
                                'arn': profile_arn,
                                'name': profile_name,
                                'status': status,
                                'models': foundation_models
                            })
                            self.logger.info(f"✅ Found compatible profile: {profile_name}")
                            break
                
                except Exception as e:
                    self.logger.debug(f"Could not get details for profile {profile_name}: {e}")
                    continue
            
            if not matching_profiles:
                self.logger.warning(f"⚠️ No inference profiles found that contain model {model_id} in {self.region}")
                
                # Try other regions if not found in primary region
                if self.region != "us-east-1":
                    self.logger.info(f"🔍 Trying other regions for inference profiles...")
                    return self._search_inference_profiles_multiple_regions(model_id)
                
                return None
            
            # Select the best profile (prefer profiles with fewer models, or the first one)
            best_profile = min(matching_profiles, key=lambda p: len(p['models']))
            
            self.logger.info(f"🎯 Selected inference profile: {best_profile['name']}")
            self.logger.info(f"   🔗 ARN: {best_profile['arn']}")
            self.logger.info(f"   📊 Contains {len(best_profile['models'])} model(s)")
            
            return best_profile['arn']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                self.logger.warning(f"⚠️ Access denied when listing inference profiles")
                self.logger.warning(f"   Check IAM permissions for bedrock:ListInferenceProfiles")
            else:
                self.logger.warning(f"⚠️ Error discovering inference profiles: {error_code}")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Could not auto-discover inference profile: {e}")
            return None
    
    def _is_model_compatible(self, target_model: str, profile_model: str) -> bool:
        """
        Check if a profile model is compatible with the target model.
        
        Args:
            target_model: The model we're looking for
            profile_model: The model in the inference profile
            
        Returns:
            True if compatible, False otherwise
        """
        # Exact match
        if target_model == profile_model:
            return True
        
        # Check for similar models (same base model, different versions)
        target_base = target_model.replace(':0', '').replace('us.', '').replace('eu.', '').replace('ap.', '')
        profile_base = profile_model.replace(':0', '').replace('us.', '').replace('eu.', '').replace('ap.', '')
        
        # For Haiku models, check if it's the same base model
        if 'claude-3-5-haiku' in target_base and 'claude-3-5-haiku' in profile_base:
            return True
        
        # For other Claude models, check if it's the same family
        if 'claude' in target_base and 'claude' in profile_base:
            # Extract the main model name (e.g., 'claude-3-5-sonnet' from 'claude-3-5-sonnet-20241022-v1')
            target_family = target_base.split('-2024')[0].split('-2023')[0] if '-' in target_base else target_base
            profile_family = profile_base.split('-2024')[0].split('-2023')[0] if '-' in profile_base else profile_base
            
            return target_family == profile_family
        
        return False
    
    def _search_inference_profiles_multiple_regions(self, model_id: str) -> Optional[str]:
        """
        Search for inference profiles across multiple regions.
        
        Args:
            model_id: The model ID that requires an inference profile
            
        Returns:
            The inference profile ARN if found, None otherwise
        """
        # Common regions to search
        regions_to_try = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        
        # Remove the current region from the list
        if self.region in regions_to_try:
            regions_to_try.remove(self.region)
        
        for region in regions_to_try:
            try:
                self.logger.info(f"🔍 Searching in region: {region}")
                
                import boto3
                session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
                temp_client = session.client('bedrock', region_name=region)
                
                # List inference profiles in this region
                response = temp_client.list_inference_profiles()
                profiles = response.get('inferenceProfileSummaries', [])
                
                if not profiles:
                    continue
                
                # Look for profiles that contain the target model
                for profile in profiles:
                    profile_arn = profile.get('inferenceProfileArn', '')
                    profile_name = profile.get('inferenceProfileName', '')
                    status = profile.get('status', '')
                    
                    if status != 'ACTIVE':
                        continue
                    
                    try:
                        # Get detailed information about the profile
                        detail_response = temp_client.get_inference_profile(inferenceProfileIdentifier=profile_arn)
                        detail = detail_response.get('inferenceProfile', {})
                        foundation_models = detail.get('foundationModelArns', [])
                        
                        # Check if this profile contains the target model
                        for model_arn in foundation_models:
                            model_id_from_arn = model_arn.split('/')[-1] if '/' in model_arn else model_arn
                            
                            if (model_id_from_arn == model_id or 
                                self._is_model_compatible(model_id, model_id_from_arn)):
                                
                                self.logger.info(f"✅ Found inference profile in {region}: {profile_name}")
                                self.logger.info(f"   🔗 ARN: {profile_arn}")
                                return profile_arn
                    
                    except Exception as e:
                        self.logger.debug(f"Could not get details for profile {profile_name} in {region}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.debug(f"Could not search region {region}: {e}")
                continue
        
        self.logger.warning(f"⚠️ No inference profiles found for model {model_id} in any region")
        return None
    
    def _log_available_models_suggestion(self):
        """Log available models to help users choose alternatives."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create a temporary client for checking model availability
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
            temp_client = session.client('bedrock', region_name=self.region)
            
            # List foundation models
            response = temp_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            
            # Find some good alternatives
            suggested_models = []
            
            # Look for Claude models that support on-demand
            claude_models = [m for m in available_models if 'claude' in m.lower() and 'haiku' not in m.lower()]
            if claude_models:
                suggested_models.extend(claude_models[:2])  # Take first 2
            
            # Look for other good models
            other_models = [m for m in available_models if 'amazon.titan' in m or 'meta.llama' in m]
            if other_models:
                suggested_models.extend(other_models[:2])  # Take first 2
            
            if suggested_models:
                self.logger.info(f"💡 Available models in {self.region} that support on-demand access:")
                for model in suggested_models[:3]:  # Show up to 3 models
                    self.logger.info(f"   • {model}")
                self.logger.info(f"   💡 You can use any of these models instead")
            else:
                self.logger.warning(f"⚠️ No obvious alternative models found in {self.region}")
                self.logger.warning(f"   Consider checking other regions or creating inference profiles")
                
        except Exception as e:
            self.logger.debug(f"Could not list available models: {e}")
    
    def _check_model_availability(self, model_id: str) -> bool:
        """
        Check if a model is available in the current region.
        
        Args:
            model_id: The model ID to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create a temporary client for checking model availability
            session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', 'default'))
            temp_client = session.client('bedrock', region_name=self.region)
            
            # List foundation models to check availability
            response = temp_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            
            # Check if model is in the list
            is_available = model_id in available_models
            
            if is_available:
                self.logger.info(f"✅ Model {model_id} is available in region {self.region}")
            else:
                self.logger.warning(f"⚠️ Model {model_id} not found in available models for region {self.region}")
                self.logger.warning(f"   Available models: {available_models[:5]}...")
            
            return is_available
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check model availability: {e}")
            return True  # Assume available if check fails
    
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
        
        # Use model-specific parameter names
        config = {
            "maxTokenCount": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "topP": kwargs.get('top_p', 0.9),
            "stopSequences": kwargs.get('stop_sequences', [])
        }
        
        # Remove None values and empty sequences
        config = {k: v for k, v in config.items() if v is not None and v != []}
        
        return {
            "inputText": prompt,
            "textGenerationConfig": config
        }
