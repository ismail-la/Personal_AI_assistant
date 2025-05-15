import wave
import os
import json
from vosk import Model, KaldiRecognizer

class SpeechToText:
    def __init__(self, model_path=None):
        # Default to the 'model' subdirectory within the same directory as this script
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model")
        
        print(f"Loading Vosk model from: {model_path}")
        try:
            self.model = Model(model_path)
            print("Vosk model loaded successfully")
        except Exception as e:
            print(f"Failed to load Vosk model: {e}")
            print("Please download a model from https://alphacephei.com/vosk/models")
            print("and extract it to the 'model' directory")
            raise

    def transcribe(self, wav_path: str) -> str:
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())
        
        text_result = ""
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result_dict = json.loads(rec.Result())
                if "text" in result_dict:
                    text_result += result_dict["text"] + " "
                    
        # Process final result
        final_result = json.loads(rec.FinalResult())
        if "text" in final_result:
            text_result += final_result["text"]
            
        return text_result.strip()