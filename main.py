# main.py
import gradio as gr
from config.logging_config import setup_logging
from storyteller.caption_story import CaptionStoryGenerator
from storyteller.interaction import InteractionHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Set up logging
setup_logging()

# Initialize LLaMA3 model with LangChain
local_llm = 'llama3'
llm = ChatOllama(model=local_llm, keep_alive="-1", max_tokens=3000, temperature=0)

# Create prompt templates
story_template = """Based solely on the provided caption, please generate a detailed and engaging story.

Caption:
{caption}

Generate a story based on the caption provided and only return story do not add any interactive text from your side"""
story_prompt = ChatPromptTemplate.from_template(story_template)

interaction_template = """Based on the provided caption and conversation history, continue the conversation.

Caption:
{caption}

Conversation History:
{history}

please do not add interactive text from your end just provide response to the query"""
interaction_prompt = ChatPromptTemplate.from_template(interaction_template)

# Initialize conversation history as a list
conversation_history = []

# Initialize CaptionStoryGenerator and InteractionHandler
caption_story_generator = CaptionStoryGenerator()
interaction_handler = InteractionHandler(llm, interaction_prompt, conversation_history)

# Function to handle image input and generate a caption and story
def process_image(image):
    return caption_story_generator.generate_caption_story(
        image=image, 
        llm=llm, 
        story_prompt=story_prompt, 
        conversation_history=conversation_history
    )

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Interactive Storyteller and Chatbot")
    with gr.Column():
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an Image")
            caption_output = gr.Textbox(label="Generated Caption")
            story_output = gr.Textbox(label="Generated Story")
        
        # Update this line to correctly call the process_image function
        image_input.change(fn=process_image, inputs=image_input, outputs=[caption_output, story_output])
        
        with gr.Row():
            user_message = gr.Textbox(label="Your Message")
            chat_response = gr.Textbox(label="LLaMA3 Response")
            user_message.submit(fn=interaction_handler.handle_interaction, inputs=user_message, outputs=chat_response)
        
    iface.launch()
