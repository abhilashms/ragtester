"""
Health check utility for all LLM providers to ensure proper API functionality.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .providers import build_llm
from .api_utils import validate_api_key


@dataclass
class ProviderHealthStatus:
    """Health status for a provider."""
    provider: str
    status: str  # "healthy", "unhealthy", "unknown"
    error_message: Optional[str] = None
    model: Optional[str] = None
    response_time: Optional[float] = None


class ProviderHealthChecker:
    """Health checker for all LLM providers."""
    
    def __init__(self):
        self.test_messages = [
            {"role": "user", "content": "Hello! Please respond with 'Health check successful.'"}
        ]
    
    def check_provider_health(self, provider_name: str, model: str = None, **kwargs) -> ProviderHealthStatus:
        """
        Check the health of a specific provider.
        
        Args:
            provider_name: Name of the provider to check
            model: Model to test (optional, uses default if not provided)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ProviderHealthStatus object with health information
        """
        import time
        
        try:
            # Build the provider
            provider = build_llm(provider_name, model=model, **kwargs)
            
            if provider is None:
                return ProviderHealthStatus(
                    provider=provider_name,
                    status="unhealthy",
                    error_message="Failed to initialize provider"
                )
            
            # Test the provider with a simple message
            start_time = time.time()
            response = provider.chat(self.test_messages)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Check if response is valid
            if response and len(response.strip()) > 0:
                return ProviderHealthStatus(
                    provider=provider_name,
                    status="healthy",
                    model=model,
                    response_time=response_time
                )
            else:
                return ProviderHealthStatus(
                    provider=provider_name,
                    status="unhealthy",
                    error_message="Empty or invalid response",
                    model=model,
                    response_time=response_time
                )
                
        except Exception as e:
            return ProviderHealthStatus(
                provider=provider_name,
                status="unhealthy",
                error_message=str(e),
                model=model
            )
    
    def check_all_providers(self, provider_configs: Dict[str, Dict[str, Any]] = None) -> List[ProviderHealthStatus]:
        """
        Check the health of all available providers.
        
        Args:
            provider_configs: Dictionary mapping provider names to their configurations
            
        Returns:
            List of ProviderHealthStatus objects
        """
        if provider_configs is None:
            provider_configs = self._get_default_provider_configs()
        
        health_statuses = []
        
        for provider_name, config in provider_configs.items():
            try:
                status = self.check_provider_health(provider_name, **config)
                health_statuses.append(status)
            except Exception as e:
                health_statuses.append(ProviderHealthStatus(
                    provider=provider_name,
                    status="unhealthy",
                    error_message=f"Health check failed: {str(e)}"
                ))
        
        return health_statuses
    
    def _get_default_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all providers."""
        configs = {}
        
        # Check for API keys and create configs for available providers
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "grok": os.getenv("XAI_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "reka": os.getenv("REKA_API_KEY"),
            "qwen": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            "moonshot": os.getenv("MOONSHOT_API_KEY"),
            "zhipu": os.getenv("ZHIPU_API_KEY"),
            "baidu": os.getenv("BAIDU_API_KEY"),
            "zeroone": os.getenv("ZEROONE_API_KEY"),
        }
        
        # Add providers with available API keys
        for provider, api_key in api_keys.items():
            if api_key:
                if provider == "openai":
                    configs[provider] = {"model": "gpt-4o-mini"}
                elif provider == "anthropic":
                    configs[provider] = {"model": "claude-3-5-sonnet-20241022"}
                elif provider == "google":
                    configs[provider] = {"model": "gemini-1.5-flash"}
                elif provider == "mistral":
                    configs[provider] = {"model": "mistral-large-latest"}
                elif provider == "cohere":
                    configs[provider] = {"model": "command-r-plus"}
                elif provider == "grok":
                    configs[provider] = {"model": "grok-beta"}
                elif provider == "deepseek":
                    configs[provider] = {"model": "deepseek-chat"}
                elif provider == "reka":
                    configs[provider] = {"model": "reka-core-20240719"}
                elif provider == "qwen":
                    configs[provider] = {"model": "qwen-plus"}
                elif provider == "moonshot":
                    configs[provider] = {"model": "moonshot-v1-8k"}
                elif provider == "zhipu":
                    configs[provider] = {"model": "glm-4"}
                elif provider == "baidu":
                    configs[provider] = {"model": "ernie-bot"}
                elif provider == "zeroone":
                    configs[provider] = {"model": "yi-34b-chat"}
        
        # Add Bedrock if AWS credentials are available
        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
            configs["bedrock"] = {
                "model": "anthropic.claude-sonnet-4-20250514-v1:0",  # Use model that supports on-demand access
                "region": "us-east-1"
            }
        
        # Add local providers (always available)
        configs["dummy"] = {}
        
        return configs
    
    def print_health_report(self, health_statuses: List[ProviderHealthStatus]) -> None:
        """Print a formatted health report."""
        from ..logging_utils import get_logger
        
        logger = get_logger()
        
        with logger.operation_context("health_report"):
            logger.info("\n" + "="*60)
            logger.info("LLM PROVIDER HEALTH CHECK REPORT")
            logger.info("="*60)
            
            healthy_count = 0
            unhealthy_count = 0
            
            for status in health_statuses:
                if status.status == "healthy":
                    healthy_count += 1
                    logger.info(f"✅ {status.provider.upper()}")
                    if status.model:
                        logger.info(f"   Model: {status.model}")
                    if status.response_time:
                        logger.info(f"   Response Time: {status.response_time:.2f}s")
                else:
                    unhealthy_count += 1
                    logger.warning(f"❌ {status.provider.upper()}")
                    if status.model:
                        logger.warning(f"   Model: {status.model}")
                    if status.error_message:
                        logger.warning(f"   Error: {status.error_message}")
            
            logger.info("\n" + "-"*60)
            logger.info(f"SUMMARY: {healthy_count} healthy, {unhealthy_count} unhealthy")
            logger.info("="*60)


def run_health_check(provider_name: str = None, **kwargs) -> ProviderHealthStatus:
    """
    Run a health check for a specific provider or all providers.
    
    Args:
        provider_name: Name of provider to check (if None, checks all)
        **kwargs: Provider-specific configuration
        
    Returns:
        ProviderHealthStatus or list of ProviderHealthStatus objects
    """
    checker = ProviderHealthChecker()
    
    if provider_name:
        return checker.check_provider_health(provider_name, **kwargs)
    else:
        return checker.check_all_providers()


if __name__ == "__main__":
    # Run health check for all providers
    checker = ProviderHealthChecker()
    health_statuses = checker.check_all_providers()
    checker.print_health_report(health_statuses)
