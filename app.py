import gradio as gr
from storyteller.image_captioner import ImageCaptioner
from storyteller.story_generator import StoryGenerator
from storyteller.conversation_manager import ConversationManager

class StorytellingApp:
    def __init__(self):
        self.image_captioner = ImageCaptioner()
        self.story_generator = StoryGenerator()
        self.conversation_manager = ConversationManager()

    def process_image(self, image):
        caption = self.image_captioner.generate(image)
        story = self.story_generator.generate(caption)
        self.conversation_manager.add_interaction([story, None])
        return caption, story

    def continue_conversation(self, user_input, caption=None):
        # self.conversation_manager.add_interaction(user_input)
        response = self.story_generator.generate(
            caption, 
            self.conversation_manager.get_history()
        )
        self.conversation_manager.add_interaction([user_input, response])
        return [[user_input, response]]
    
def main():
    app = StorytellingApp()
    
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Storytelling Assistant")
        
        with gr.Row():
            # Left column for image upload and caption
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                submit_btn = gr.Button("Generate Story", variant="primary")
                caption_output = gr.Textbox(
                    label="Image Caption",
                    lines=2,
                    interactive=False
                )
            
            # Right column for story and chat
            with gr.Column(scale=2):
                story_output = gr.Textbox(
                    label="Generated Story",
                    lines=4,
                    interactive=False
                )
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    bubble_full_width=False,
                    show_label=True
                )
                txt_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                chat_btn = gr.Button("Send Message", variant="secondary")

        def process_image(image):
            if image is None:
                return "Please upload an image first.", "No story generated.", []
            caption, story = app.process_image(image)
            history = app.conversation_manager.get_history()
            return caption, story, history

        def send_message(user_message):
            try:
                if not user_message:
                    return []
                history = app.conversation_manager.get_history()
                if not history:
                    # First message; pass the story
                    response = app.continue_conversation(user_message, story_output)
                else:
                    # Subsequent messages; use history
                    response = app.continue_conversation(user_message)
                history = app.conversation_manager.get_history()
                return history
            except Exception as e:
                print(f"Error in send_message: {e}")
                return []



        submit_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[caption_output, story_output, chatbot]
        )

        txt_input.submit(
            fn=send_message,
            inputs=[txt_input],
            outputs=[chatbot]
        )
        chat_btn.click(
            fn=send_message,
            inputs=[txt_input],
            outputs=[chatbot]
        )
    
    interface.launch(share=True)

if __name__ == "__main__":
    main()