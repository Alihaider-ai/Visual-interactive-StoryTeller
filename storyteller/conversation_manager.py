class ConversationManager:
    def __init__(self):
        self.history = []

    def add_interaction(self, message):
        self.history.append(message)

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []