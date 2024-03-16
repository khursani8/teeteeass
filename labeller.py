import gradio as gr
import fire

g_texts = []
g_arpas = []
g_path = None

def b_change_index(index,a=None,b=None):
    index = int(index)
    return g_texts[index],g_arpas[index]

def b_submit_change(index,text,arpa):
    index = int(index)
    g_texts[index] = text
    g_arpas[index] = arpa
    with open(g_path,'w', encoding="utf-8") as file:
        for idx in range(len(g_texts)):
            text = g_texts[idx]
            arpa = g_arpas[idx]
            file.write(f"{text}|{arpa}".strip()+'\n')
    index += 1
    return index,g_texts[index],g_arpas[index]

def start(path):
    global g_path
    g_path = path
    global g_texts
    texts = g_texts
    global g_arpas
    arpas = g_arpas
    lines = open(path).readlines()
    for line in lines:
        text, arpa = line.split("|")
        texts.append(text)
        arpas.append(arpa)

    print()
    with gr.Blocks() as gui:
        indexbox = gr.Textbox(
            label = "Index",
            visible = True,
            scale=5,
            value=0
        )
        with gr.Column():
            with gr.Row():
                textbox = gr.Textbox(
                    label = "Text",
                    visible = True,
                    scale=5
                )
                arpabox = gr.Textbox(
                    label = "Arpa",
                    visible = True,
                    scale=5
                )
        btn_submit_change = gr.Button('Submit')

        btn_submit_change.click(
            b_submit_change,
            inputs=[
                indexbox,
                textbox,
                arpabox,
            ],
            outputs=[
                indexbox,
                textbox,
                arpabox
            ],
        )
        gui.load(
            b_change_index,
            inputs=[
                indexbox,
                textbox,
                arpabox
            ],
            outputs=[
                textbox,
                arpabox
            ],
        )


    gui.launch(server_name='0.0.0.0', server_port=7860)

if __name__ == "__main__":
    fire.Fire(start)