import gradio as gr
import subprocess

def run_task(task):
    process = subprocess.Popen(
        ["python", "inference.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = ""
    for line in process.stdout:
        if f'"task_id": "{task}"' in line or "[STEP]" in line or "[END]" in line:
            output += line

    return output


demo = gr.Interface(
    fn=run_task,
    inputs=gr.Dropdown(
        ["simple-fill", "adaptive-execution", "multi-asset"],
        label="Select Task"
    ),
    outputs=gr.Textbox(lines=25),
    title="🚀 Trading Execution Agent",
    description="TWAP + Adaptive Execution Strategy"
)

demo.launch()