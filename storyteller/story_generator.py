from .base import StorytellerBase
from config.model_config import ModelLoader,ModelConfig
from config.prompt_config import PromptTemplates

class StoryGenerator(StorytellerBase):
    def __init__(self):
        # self.llm = ModelLoader.load_llm()
        config = ModelConfig(llm_type="gemini")
        self.llm = ModelLoader.load_llm(config)

        self.story_prompt = PromptTemplates.get_story_prompt()
        self.interaction_prompt = PromptTemplates.get_interaction_prompt()

    def generate(self, caption, history=None):
        if history is None:
            chain = self.story_prompt | self.llm
            return chain.invoke({"caption": caption}).content
        
        chain = self.interaction_prompt | self.llm
        return chain.invoke({
            "history": history
        }).content