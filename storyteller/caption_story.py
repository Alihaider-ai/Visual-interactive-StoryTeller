# storyteller/caption_story.py
import logging
import traceback
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

class CaptionStoryGenerator:
    def __init__(self):
        try:
            logger.info("Initializing BLIP processor and model")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        except Exception as e:
            logger.error("Failed to initialize BLIP processor or model")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            raise

    def generate_caption_story(self, image, llm, story_prompt, conversation_history):
        try:
            logger.info("Generating caption from image")
            inputs = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values=inputs, max_length=50)
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Generated caption: {caption}")
            logger.info("Generating story based on the caption")
            story = llm.invoke(story_prompt.format(caption=caption)).content
            
            conversation_history.clear()
            conversation_history.append(f"Caption: {caption}")
            conversation_history.append(f"Story: {story}")
            
            logger.info("Story generated successfully")
            return caption, story
        except Exception as e:
            logger.error("Error occurred during caption or story generation")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            raise
