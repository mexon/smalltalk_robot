from whisper_mic import WhisperMic
from plan import VoiceStateMachine
import pyttsx3

tts_engine = pyttsx3.init()
whisper_mic = WhisperMic(english=True, implementation="whisper")

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
            "no": {"new_state": "no_tour", "persona": "Mei-Hua"},
        },
    },
    "no_tour": {
        "prompt": "No problem, have a great visit!",
        "new_state": "start",
    },
    "introduction": {
        "prompt": "Great, let's get started, please come this way to the first exhibit.",
        "new_state": "weather_prompt",
    },
    "weather_prompt": {
        "prompt": "It's a nice day don't you think?",
        "new_state": "weather_response",
    },
    "weather_response": {
        "responses": {
            "no": {"new_state": "snake_travel", "persona": "Daniel"},
            "thank": {"new_state": "snake_travel", "persona": "Yu-Hsuan"},
        },
    },
    "snake_travel": {
        "buttons": {
            "a": {"new_state": "snake_description"},
        },
    },
    "snake_description": {
        "prompt": "Here we are at the first exhibit.  This is an articulated snake made of iron.  Now, let's go to the second exhibit.",
        "new_state": "sport_prompt",
    },
    "sport_prompt": {
        "prompt": "Did you see the game last night?",
        "new_state": "sport_response",
    },
    "sport_response": {
        "responses": {
            "yes": {"new_state": "sport_extra", "persona": "Yu-Hsuan"},
            "no": {"new_state": "sword_travel", "persona": "Daniel"},
        },
    },
    "sport_extra": {
        "prompt": "Yes, I like sports too.",
        "new_state": "sword_travel",
    },
    "sword_travel": {
        "buttons": {
            "a": {"new_state": "sword_description"},
        },
    },
    "sword_description": {
        "prompt": "This sword looks like it's made out of steel, but it's actually papier maché.  This concludes our tour.  Bye-bye!",
        "new_state": "start",
    },
}

state_machine = VoiceStateMachine(plan, tts_engine, whisper_mic)
while state_machine.step():
    state_machine.print_state()
