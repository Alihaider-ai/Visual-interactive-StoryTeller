import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

# Function to generate a caption and story
def generate_caption_story(image):
    # Generate caption from image using BLIP
    inputs = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=inputs, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Generate story based on the caption using LLaMA3
    story = llm.invoke(story_prompt.format(caption=caption)).content
    
    # Initialize conversation history with the caption and story
    conversation_history.clear()
    conversation_history.append(f"Caption: {caption}")
    conversation_history.append(f"Story: {story}")
    
    return caption, story

# Function to handle chat interaction
def handle_interaction(user_message):
    # Add the user message to the conversation history
    conversation_history.append(f"User: {user_message}")
    
    # Generate LLaMA3 response to the user message
    response = llm.invoke(interaction_prompt.format(caption=conversation_history[0].split(': ')[1], history='\n'.join(conversation_history))).content
    
    # Add the LLaMA3 response to the conversation history
    conversation_history.append(f"LLaMA3: {response}")
    
    return response

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Interactive Storyteller and Chatbot")
    with gr.Column():
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an Image")
            caption_output = gr.Textbox(label="Generated Caption")
            story_output = gr.Textbox(label="Generated Story")
        image_input.change(fn=generate_caption_story, inputs=image_input, outputs=[caption_output, story_output])
        
        with gr.Row():
            user_message = gr.Textbox(label="Your Message")
            chat_response = gr.Textbox(label="LLaMA3 Response")
            user_message.submit(fn=handle_interaction, inputs=user_message, outputs=chat_response)
        
    iface.launch()

