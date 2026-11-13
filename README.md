# Smalltalk Museum Robot

This collects documents for a small project to investigate using smalltalk to enhance the usefulness of a museum tour-guide robot.

## Whisper Proof of Concept

There is a proof-of-concept script for voice interaction with classification.  It does not yet connect to a LLM.

    $ python3 -m venv .venv
    $ . .venv/bin/activate
    (.venv) $ pip3 install -r requirements.txt 
    (.venv) $ python3 whisper_poc.py

Press `v` to signal that a visitor has arrived.  Press `a` to signal that we have arrived at an exhibit.  Press `q` to quit.

