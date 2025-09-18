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
        self.model_id = model or "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        self.region = region or "us-east-1"
        self.kwargs = kwargs
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock client."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Initialize Bedrock client
            self._client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                **self.kwargs
            )
            
            # Test the connection by listing available models
            try:
                bedrock_client = boto3.client('bedrock', region_name=self.region)
                response = bedrock_client.list_foundation_models()
                available_models = [model['modelId'] for model in response['modelSummaries']]
                
                if self.model_id not in available_models:
                    print(f"Warning: Model {self.model_id} may not be available in region {self.region}")
                    print(f"Available models: {available_models[:5]}...")  # Show first 5 models
                    
            except ClientError as e:
                print(f"Warning: Could not verify model availability: {e}")
                
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock. "
                "Install with: pip install boto3"
            )
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Bedrock model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if self._client is None:
            raise RuntimeError("Bedrock client not initialized")
        
        # Convert messages to the format expected by the model
        formatted_messages = self._format_messages(messages)
        
        # Determine the request body format based on model
        if "anthropic" in self.model_id.lower():
            request_body = self._create_anthropic_request(formatted_messages, kwargs)
        else:
            # Generic format for other models
            request_body = self._create_generic_request(formatted_messages, kwargs)
        
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text based on model type
            if "anthropic" in self.model_id.lower():
                return response_body['content'][0]['text']
            elif "amazon" in self.model_id.lower():
                return response_body['generation']
            else:
                # Generic extraction
                return response_body.get('outputs', [{}])[0].get('text', '')
                
        except Exception as e:
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
