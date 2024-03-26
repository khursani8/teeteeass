import gradio as gr
from teeteeass.infer import TTS

def generate_tts(text, language, speaker):
    tts = TTS(speaker=speaker)
    audio = tts.generate(text, language)
    print(audio.shape)
    return 44100,audio

# Fetch the list of speakers
speakers = TTS.list_speakers()

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_tts,
    inputs=[
        gr.Textbox(lines=5, placeholder="Type your text here..."), 
        gr.Radio(choices=["EN", "ZH", "JP"], label="Language"),
        gr.Dropdown(choices=speakers, label="Speaker")  # Dropdown for speaker selection
    ],
    outputs=[gr.Audio(type="numpy", label="Generated Speech")],
    title="TTS Generation",
    description="Generate speech from text using TTS. Select your preferred language and speaker."
)

# Run the Gradio app
iface.launch()
