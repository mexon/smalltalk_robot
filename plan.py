from console.utils import wait_key

class VoiceStateMachine():
    def __init__(self, plan, tts_engine, whisper_mic):
        self.plan = plan
        self.tts_engine = tts_engine
        self.whisper_mic = whisper_mic
        self.current_state = "start"
        self.persona = None
        self.last_state = None
        self.last_prompt = None
        self.last_response = None
        self.last_button = None

    def step(self):
        self.last_state = self.current_state
        state = self.plan[self.current_state]
        if "prompt" in state:
            self.speak_prompt(state["prompt"])
        if "responses" in state:
            self.listen_to_response(state["responses"])
        if "buttons" in state:
            self.wait_for_button(state["buttons"])
        if "new_state" in state:
            self.current_state = state["new_state"]
        return self.current_state != "end"

    def speak_prompt(self, prompt):
        self.last_prompt = prompt
        self.tts_engine.say(prompt)
        self.tts_engine.runAndWait()

    def listen_to_response(self, responses):
        while True:
            if self.listen_to_one_response(responses):
                return

    def listen_to_one_response(self, responses):
        self.print_options(responses.keys())
        self.last_response = self.whisper_mic.listen()
        for response, action in responses.items():
            if response in self.last_response.lower():
                self.perform_action(action)
                return True

    def wait_for_button(self, buttons):
        self.print_options(buttons.keys())
        self.last_button = wait_key()
        if self.last_button == 'q':
            exit(0)
        for button, action in buttons.items():
            if self.last_button == button:
                self.perform_action(action)
                return True

    def print_options(self, options):
        print(f"Waiting for: {", ".join(options)}")

    def perform_action(self, action):
        if "new_state" in action:
            self.current_state = action["new_state"]
        if "persona" in action:
            self.persona = action["persona"]
