import gradio as gr
import time
from index import whisper_fn
import os
import whisper





import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



os.environ["OPENAI_API_KEY"] = "xyz"

def transcribe(audio):
    # # time.sleep(3)
    # # load audio and pad/teim it to fit 30 seconds
    # audio = whisper.load(audio)
    # audio = whisper.pad_or_trim(audio)

    # # make log-mel  spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # #detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(prob, key-probs.get)}")

    # # decode the audio
    # option = whisper.DecodingOptions()
    # result = whisper.decode(model, mel, option)
    res = whisper_fn(audio)
    
    res_instructed = "Rate the answer to the question as RED, AMBER or GREEN where RED is wrong, Amber is relevant and Green is correct answer\n" + res
    chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
    response_gpt = chatgpt([HumanMessage(content=res_instructed)])
    output = "Transcription: \n" + res + "\n\n RAG Score:\n" + response_gpt.content
    return output


gr.Interface(
    title = 'Real-time AI-base Audio Transcription, Recognition and Translation web App',
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()