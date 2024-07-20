# Interactive Storyteller and Chatbot

This project uses Gradio to create an interactive interface where users can upload an image, generate a caption and story based on the image using BLIP and LLaMA3 models, and then interact with the story through a chatbox.

## Features

- **Image Captioning**: Upload an image to generate a caption using the BLIP model.
- **Story Generation**: Generate a detailed and engaging story based on the caption using the LLaMA3 model.
- **Interactive Chat**: Engage in a conversation with the chatbot, which maintains context and conversation history.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Install Required Packages

Clone the repository:

```bash
git clone https://https://github.com/Alihaider-ai/Visual-interactive-StoryTeller.git
cd interactive-storyteller-chatbot
```
### Required Packages:

```bash
gradio
transformers
torch
langchain
langchain_community
pillow
```

### Usage
Run the application:

```bash
python app.py
```
Open the Gradio interface in your web browser (usually at http://127.0.0.1:7860).

- Upload an image to generate a caption and story.
- Use the chatbox to interact with the generated story.

## Code Explanation

### `app.py`

The main script initializes the BLIP processor and model for image captioning, and the LLaMA3 model for generating stories and interacting with users. It uses Gradio to create a user-friendly interface where:

- Users can upload an image to generate a caption and story.
- A chatbox interface allows users to interact with the generated story.

### Functions

- **`generate_caption_story`**: Handles the image upload, generating the caption and the initial story.
- **`handle_interaction`**: Handles the chat interaction, maintaining conversation history for context.

### Image of Working

![image](https://github.com/user-attachments/assets/b483c3a4-4641-4be1-89de-7b67e8440a4c)




## Acknowledgements

- Gradio for the interactive UI framework.
- Hugging Face for the BLIP model.
- LangChain and Ollama for managing the LLaMA3 model integration.

