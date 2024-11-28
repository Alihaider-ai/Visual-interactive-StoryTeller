from langchain.prompts import ChatPromptTemplate

class PromptTemplates:
    STORY_TEMPLATE = """Based solely on the provided caption, please generate a detailed and engaging story.

Caption:
{caption}

Generate a story based on the caption provided and only return story do not add any interactive text from your side"""

    INTERACTION_TEMPLATE = """Based on the provided story(in history) and conversation history, continue the conversation.


Conversation History:
{history}

please do not add interactive text from your end just provide response to the query"""

    @classmethod
    def get_story_prompt(cls):
        return ChatPromptTemplate.from_template(cls.STORY_TEMPLATE)

    @classmethod
    def get_interaction_prompt(cls):
        return ChatPromptTemplate.from_template(cls.INTERACTION_TEMPLATE)