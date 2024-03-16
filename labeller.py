import gradio as gr
import fire

class PhonemeAnnotator:
    def __init__(self, path):
        self.path = path
        self.texts = []
        self.arpas = []
        self.load_data()

    def load_data(self):
        try:
            with open(self.path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    text, arpa = line.strip().split("|")
                    self.texts.append(text)
                    self.arpas.append(arpa)
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            exit()

    def save_data(self):
        with open(self.path, 'w', encoding="utf-8") as file:
            for text, arpa in zip(self.texts, self.arpas):
                file.write(f"{text}|{arpa}\n")

    def submit_change(self, index, text, arpa):
        index = int(index)
        if index < 0 or index >= len(self.texts):
            return "Invalid index", "", ""
        self.texts[index] = text
        self.arpas[index] = arpa
        self.save_data()
        next_index = min(index + 1, len(self.texts) - 1)
        return str(next_index), self.texts[next_index], self.arpas[next_index]

    def change_index(self, index):
        index = int(index)
        if index < 0 or index >= len(self.texts):
            return "", ""
        return self.texts[index], self.arpas[index]

    def launch_gui(self):
        with gr.Blocks() as gui:
            with gr.Row():
                with gr.Column():
                    indexbox = gr.Textbox(label="Index", value="0")
                    textbox = gr.Textbox(label="Text")
                    arpabox = gr.Textbox(label="Arpa")
                    btn_submit_change = gr.Button('Submit')

                    btn_submit_change.click(
                        self.submit_change,
                        inputs=[indexbox, textbox, arpabox],
                        outputs=[indexbox, textbox, arpabox]
                    )

                    gui.load(
                        self.change_index,
                        inputs=[indexbox],
                        outputs=[textbox, arpabox]
                    )
                with gr.Column():  # Enable scrolling for longer content
                    gr.Markdown("""
    ### ARPAbet Reference Guide

    Here's a comprehensive list of ARPAbet phonemes, their examples, and phonetic transcriptions:

    | Phoneme | Example | Translation |
    | ------- | ------- | ----------- |
    | AA      | odd     | AA D        |
    | AE      | at      | AE T        |
    | AH      | hut     | HH AH T     |
    | AO      | ought   | AO T        |
    | AW      | cow     | K AW        |
    | AY      | hide    | HH AY D     |
    | B       | be      | B IY        |
    | CH      | cheese  | CH IY Z     |
    | D       | dee     | D IY        |
    | DH      | thee    | DH IY       |
    | EH      | Ed      | EH D        |
    | ER      | hurt    | HH ER T     |
    | EY      | ate     | EY T        |
    | F       | fee     | F IY        |
    | G       | green   | G R IY N    |
    | HH      | he      | HH IY       |
    | IH      | it      | IH T        |
    | IY      | eat     | IY T        |
    | JH      | gee     | JH IY       |
    | K       | key     | K IY        |
    | L       | lee     | L IY        |
    | M       | me      | M IY        |
    | N       | knee    | N IY        |
    | NG      | ping    | P IH NG     |
    | OW      | oat     | OW T        |
    | OY      | toy     | T OY        |
    | P       | pee     | P IY        |
    | R       | read    | R IY D      |
    | S       | sea     | S IY        |
    | SH      | she     | SH IY       |
    | T       | tea     | T IY        |
    | TH      | theta   | TH EY T AH  |
    | UH      | hood    | HH UH D     |
    | UW      | two     | T UW        |
    | V       | vee     | V IY        |
    | W       | we      | W IY        |
    | Y       | yield   | Y IY L D    |
    | Z       | zee     | Z IY        |
    | ZH      | seizure | S IY ZH ER  |
    """)
        gui.launch(server_name='0.0.0.0', server_port=7860)

def start(path):
    annotator = PhonemeAnnotator(path)
    annotator.launch_gui()

if __name__ == "__main__":
    fire.Fire(start)
