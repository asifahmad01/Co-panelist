import os
import whisper
import gradio as gr
import time



# def transcribe(audio):
    
#     model = whisper.load_model("base")
#     model.device
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)

    
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

    
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")


#     options = whisper.DecodingOptions(fp16 = False)
#     result = whisper.decode(model, mel, options)
#     return result.text


# gr.Interface(
#     title = 'Real-time AI-based Audio Transcription, Recognition and Translation Web App', 
#     fn=transcribe, 
#     inputs=[
#         gr.inputs.Audio(source="microphone", type="filepath")
#     ],
#     outputs=[
#         "textbox"
#     ],
#     live=True).launch(share=True)


def whisper_fn(audio):
    model = whisper.load_model("base")
    model.device

        # from IPython.display import Audio
        # Audio("sample.mp3")

        # from IPython.display import Audio
        # Audio("./trail.mp3")

        # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

        # make log-mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

        #detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
    option = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, option)

        # print the recognized text
    return result.text