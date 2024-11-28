import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

ModelType = Literal["ollama", "gemini"]

@dataclass
class ModelConfig:
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    llm_type: ModelType = "ollama"
    llm_config: dict = None
    temperature: float = 0
    max_tokens: int = 3000

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = {
                "ollama": {"model": "llama3"},
                "gemini": {"model": "gemini-pro", "google_api_key": os.getenv("GOOGLE_API_KEY")}
            }

class LLMFactory:
    @staticmethod
    def create_llm(config: ModelConfig):
        if config.llm_type == "ollama":
            return ChatOllama(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                keep_alive="-1",
                **config.llm_config["ollama"]
            )
        elif config.llm_type == "gemini":
            return ChatGoogleGenerativeAI(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
                **config.llm_config["gemini"]
            )
        raise ValueError(f"Unsupported model type: {config.llm_type}")

class ModelLoader:
    @staticmethod
    def load_blip():
        processor = BlipProcessor.from_pretrained(ModelConfig.blip_model_name)
        model = BlipForConditionalGeneration.from_pretrained(ModelConfig.blip_model_name)
        return processor, model

    @staticmethod
    def load_llm(config: ModelConfig):
        return LLMFactory.create_llm(config)