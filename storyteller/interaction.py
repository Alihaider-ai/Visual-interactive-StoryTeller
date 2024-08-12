import logging
import traceback

logger = logging.getLogger(__name__)

class InteractionHandler:
    def __init__(self, llm, interaction_prompt, conversation_history):
        self.llm = llm
        self.interaction_prompt = interaction_prompt
        self.conversation_history = conversation_history

    def handle_interaction(self, user_message):
        try:
            logger.info("Handling user interaction")
            self.conversation_history.append(f"User: {user_message}")
            
            logger.info("Generating response based on conversation history")
            response = self.llm.invoke(
                self.interaction_prompt.format(
                    caption=self.conversation_history[0].split(': ')[1], 
                    history='\n'.join(self.conversation_history)
                )
            ).content
            
            self.conversation_history.append(f"LLaMA3: {response}")
            
            logger.info("Response generated successfully")
            return response
        except Exception as e:
            logger.error("Error occurred during interaction handling")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            raise
