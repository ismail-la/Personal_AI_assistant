import os
import sys
import time
import sounddevice as sd
import soundfile as sf
import requests

# Add the parent directory to path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now import local modules
from voice_interface.stt import SpeechToText
from voice_interface.tts import TextToSpeech

# Define model path relative to this file
model_path = os.path.join(current_dir, "model")

try:
    print(f"Initializing SpeechToText with model path: {model_path}")
    stt = SpeechToText(model_path)
    tts = TextToSpeech()

    API_URL = "http://127.0.0.1:8000/chat/"

    def record_audio(duration=10, samplerate=16000):
        """Record audio at specified sample rate for better speech recognition"""
        print("Say something in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        
        print("🎤 Recording... (speak now, press Ctrl+C when finished)")
        try:
            # Record with a lower sample rate that Vosk prefers (16kHz)
            data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
            sd.wait()
            print("✅ Recording complete!")
            return data, samplerate
        except KeyboardInterrupt:
            sd.stop()
            print("✅ Recording stopped early.")
            return data, samplerate

    def record_and_respond():
        # Record audio
        data, samplerate = record_audio()
        
        # Save to the same directory as this script
        output_path = os.path.join(current_dir, "input.wav")
        sf.write(output_path, data, samplerate)
        
        print(f"🔄 Transcribing audio...")
        text = stt.transcribe(output_path)
        
        # Check if transcription is empty or too short
        if not text.strip() or len(text.split()) < 2:
            print(f"⚠️ Detected text too short or unclear: \"{text}\"")
            tts.speak("I couldn't understand that clearly. Could you please try again?")
            return
            
        print(f"👤 You said: \"{text}\"")
        
        # Ask for confirmation
        print("Is this correct? (y/n): ")
        confirm = input().lower().strip()
        if confirm != 'y' and confirm != 'yes':
            print("Let's try again.")
            return
        
        try:
            print("🤖 AI is thinking...")
            # Add context to the prompt to get better responses
            enhanced_prompt = f"Please answer this question clearly and directly: {text}"
            resp = requests.post(API_URL, json={"prompt": enhanced_prompt}).json()["response"]
            print(f"🤖 AI: {resp}")
            tts.speak(resp)
        except Exception as e:
            error_msg = f"❌ Error communicating with API: {str(e)}"
            print(error_msg)
            tts.speak("Sorry, I couldn't connect to the AI service.")

    if __name__ == '__main__':
        print("="*50)
        print("🎙️  Voice Assistant Started  🎙️")
        print("="*50)
        
        while True:
            try:
                record_and_respond()
                # Ask if user wants to continue
                print("\nPress Enter to ask another question or type 'exit' to quit")
                user_input = input()
                if user_input.lower() == 'exit':
                    print("\nExiting voice assistant. Goodbye!")
                    break
            except KeyboardInterrupt:
                print("\nExiting voice assistant. Goodbye!")
                break
        
except Exception as e:
    print(f"Error initializing voice interface: {str(e)}")