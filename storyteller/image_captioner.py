from .base import StorytellerBase
from config.model_config import ModelLoader

class ImageCaptioner(StorytellerBase):
    def __init__(self):
        self.processor, self.model = ModelLoader.load_blip()

    def generate(self, image):
        inputs = self.processor(image, return_tensors="pt")
        output = self.model.generate(**inputs,max_new_tokens=50)
        return self.processor.decode(output[0], skip_special_tokens=True)