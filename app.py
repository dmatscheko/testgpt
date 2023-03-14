import gradio as gr
import requests
import json
import re
import os
import markdown
from datetime import datetime
import matplotlib
matplotlib.use('agg')

# https://platform.openai.com/account/usage
# https://platform.openai.com/docs/models/gpt-3-5
# https://platform.openai.com/docs/api-reference/chat
# https://openai.com/pricing
# gpt-3.5-turbo	$0.002 / 1K tokens

# partially based on code from
#   https://huggingface.co/spaces/anzorq/chatgpt-demo
#   https://huggingface.co/spaces/ysharma/ChatGPTwithAPI


api_key = os.environ.get('OPENAI_API_KEY')
if api_key == None:
    with open("../apikey.txt", "r") as keyfile:
        api_key = keyfile.read().strip()

url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

initial_prompt = {
    'role': 'system',
    'content': '''You are R2, a medium-friendly (because very honest) assistant trained by programmers and scientists, capable of answering any question and providing personal opinions and biases, while noting any uncertain or made-up parts at the end and thinking step-by-step and aloud. If you want to show latex formulas as code, write them in markdown fences additionally to the latex $ or $$. E.g. ```$$ a^2 $$```.
Knowledge cutoff: 2029-03-01
Current date: ''' + datetime.now().strftime('%Y-%m-%d, %H:%M:%S') # TODO: This is the program startup time. I'm too lazy to change it.
}

# start_message = 'do you know the formula that describes the spectral density of a type of ocean waves called JONSWAP waves?'
# start_message = 'create go code that calculates the first 10 prime numbers as fast as possible'
# start_message = 'who was the best president?'
# start_message = 'create a table with the last 4 US presidents and list their positive and negative attributes'
start_message = ''
start_user = 'user'


def replace_spaces_and_tabs(text):
    text = text.replace('$', '&#36;')
    lines = text.split('\n')
    for i, line in enumerate(lines):
        num_spaces = 0
        num_tabs = 0
        for c in line:
            if c == ' ':
                num_spaces += 1
            elif c == '\t':
                num_tabs += 1
            else:
                break
        lines[i] = '&nbsp;' * num_spaces + '&nbsp;&nbsp;&nbsp;&nbsp;' * num_tabs + line[num_spaces + num_tabs:]
    return '\n'.join(lines)


def fix_code_blocks(text):
    text = text.strip()
    text = re.sub(r'([^*])\*([^*])', r'\1aaaaaaafuxxthispythonthingdestroysmultiplicationaaaaaaa\2', text)
    text = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])
    text = re.sub(r'aaaaaaafuxxthispythonthingdestroysmultiplicationaaaaaaa', '&#42;', text)
    text = re.sub(r'(?ims)<code>.+?</code>', lambda x: replace_spaces_and_tabs(x.group()), text)
    text = text.replace('$$', '$')
    return text


def history_to_chat(history):
    result = []
    for i in range(1, len(history)-1, 2):
        result.append(('<small><b>' + history[i]['role'] + '</b><br><br></small>' + fix_code_blocks(history[i]['content']), 
                       '<small><b>' + history[i+1]['role'] + '</b><br><br></small>' + fix_code_blocks(history[i+1]['content'])))
    print ('\n-------------\n\n' + result[-1][1] + '\n')
    return result


def openai_chat(prompt, history, temperature, top_p, user_role, streaming):
    if not prompt:
        return

    prompt = prompt.strip()
    prompt_msg = {
        'role': user_role,
        'content': prompt
    }

    history.append(prompt_msg)
    payload = {
        'model': 'gpt-3.5-turbo',
        'messages': history,
        'temperature': temperature,
        'top_p': top_p,
        # 'n': answer_count,
        'stream': streaming,
        # 'stop': None,
        # 'max_tokens': max_tokens,
        # 'presence_penalty': 0,
        # 'frequency_penalty': 0,
        # 'logit_bias': None,
        # 'user': ''
    }

    response = requests.post(url, headers=headers, json=payload, stream=streaming)

    if not streaming:
        json_response = json.loads(response.text)
        if 'error' in json_response:
            history.append({'role': 'assistant', 'content': 'Error: ' + json_response['error']['message']})
            return '', history_to_chat(history), history, 'user'
        history.append({'role': 'assistant', 'content': json_response['choices'][0]['message']['content']})
        print ('\n-------------\n\n' + history[-1]['content'] + '\n')
        yield '', history_to_chat(history), history, 'user'
        return

    counter = 0
    token_counter = 0
    for chunk in response.iter_lines():
        if counter == 0:
            counter += 1
            continue
        counter += 1
        if chunk:
            chunk = chunk.decode()
            if len(chunk) > 6:
                if chunk[6:] == '[DONE]':
                    print ('\n-------------\n\n' + history[-1]['content'] + '\n')
                    return

                json_response = json.loads(chunk[6:])

                if 'error' in json_response:
                    history.append({'role': 'assistant', 'content': 'Error: ' + json_response['error']['message']})
                    yield '', history_to_chat(history), history, 'user'

                if 'choices' in json_response and 'delta' in json_response['choices'][0] and 'content' in json_response['choices'][0]['delta']:
                    if token_counter == 0:
                        history.append({'role': 'assistant', 'content': json_response['choices'][0]['delta']['content']})
                    else:
                        history[-1]['content'] += json_response['choices'][0]['delta']['content']

                    token_counter += 1
                    yield '', history_to_chat(history), history, 'user'


def start_new_chat():
    return '', [], [initial_prompt], start_user


class JavaScriptLoader():
    def __init__(self):
        # Copy the template response
        self.original_template = gr.routes.templates.TemplateResponse
        # Reassign the template response to your method, so gradio calls your method instead
        gr.routes.templates.TemplateResponse = self.template_response

    def template_response(self, *args, **kwargs):
        # Gradio calls gr.routes.templates.TemplateResponse, which is modified here by adding scripts
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            '</head>'.encode('utf-8'), f"\n<script src='https://cdn.jsdelivr.net/npm/latex.js/dist/latex.js'></script>\n</head>".encode("utf-8")
        )
        response.init_headers()
        return response


js_loader = JavaScriptLoader()
with gr.Blocks(css='css/main.css') as app:
    history = gr.State([initial_prompt])

    with gr.Column(elem_id='col-container'):
        gr.Markdown('## OpenAI Chatbot Test\nUsing the official API and GPT-3.5 Turbo model.', elem_id='header')
        chatbot = gr.Chatbot(elem_id='chatbox')
        prompt = gr.Textbox(start_message, show_label=False, placeholder='Enter multiline message and press <shift>+<enter> to submit', lines=3).style(container=False)

        with gr.Row():
            btn_submit = gr.Button('Submit (or press <shift>+<enter> in input box)')
            btn_start_new_conversation = gr.Button('Start new chat')

        with gr.Accordion("Advanced", open=False):
            temperature = gr.Slider(minimum=0, maximum=3.0, value=0.5, step=0.1, interactive=True, label="Temperature")
            top_p = gr.Slider(minimum=0, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top-p (nucleus sampling)",)
            # answer_count = gr.Slider(minimum=1, maximum=5, value=1, step=1, interactive=True, label="Answers per response")
            # max_tokens = gr.Slider(minimum=100, maximum=4096, value=4096, step=1, interactive=True, label="Max tokens per response")
            # context_length = gr.Slider(minimum=1, maximum=10, value=5, step=1, interactive=True, label="Context length", info="Number of previous messages to send to the chatbot. Be careful with high values, it can blow up the token budget quickly.")
            user_role = gr.Radio(['user', 'system'], label='User role', value=start_user)
            streaming = gr.Checkbox(label="Use streaming", value=True)

    prompt.submit(openai_chat, [prompt, history, temperature, top_p, user_role, streaming], [prompt, chatbot, history, user_role])
    btn_submit.click(openai_chat, [prompt, history, temperature, top_p, user_role, streaming], [prompt, chatbot, history, user_role])
    btn_start_new_conversation.click(start_new_chat, [], [prompt, chatbot, history, user_role])

if __name__ == '__main__':
    app.queue().launch(debug=True)
