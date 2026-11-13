from whisper_mic import WhisperMic
from plan import VoiceStateMachine
import pyttsx3

tts_engine = pyttsx3.init()
whisper_mic = WhisperMic(english=True, implementation="faster_whisper")

plan = {
    "start": {
        "buttons": {
            "v": {"new_state": "visitor_arrived"},
        },
    },
    "visitor_arrived": {
        "prompt": "Hello, I am a museum robot, would you like a tour?",
        "new_state": "invitation_response",
    },
    "invitation_response": {
        "responses": {
            "yes": {"new_state": "introduction", "persona": "Yu-Hsuan"},
            "no": {"new_state": "idle", "persona": "Mei-Hua"},
        },
    },
    "introduction": {
        "prompt": "I am not programmed for this.",
        "new_state": "end",
    },
}

state_machine = VoiceStateMachine(plan, tts_engine, whisper_mic)
while state_machine.step():
    print(f"previous: {state_machine.last_state}, next: {state_machine.current_state}")
