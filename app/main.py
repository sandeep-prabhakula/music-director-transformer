# app/main.py
from flask import Flask, request, jsonify
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch

app = Flask(__name__)

# Load the model
model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
model.set_generation_params(duration=25)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    try:
        # Get prompts from the request
        prompts = request.json.get('prompts', [])
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400
        
        # Generate music
        res = model.generate(prompts, progress=True)
        
        # Save the generated music to files
        filenames = []
        for i, audio in enumerate(res):
            filename = f"output_{i}.wav"
            torch.save(audio, filename)
            filenames.append(filename)
        
        return jsonify({"filenames": filenames}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_music', methods=['GET'])
def cron():
    print("cron job")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
