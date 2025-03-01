from flask import Flask, request, jsonify
import openai
import whisper
import cv2
import pytesseract
from moviepy.editor import VideoFileClip
import os

app = Flask(__name__)

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Create upload directory
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to transcribe audio
def transcribe_audio(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["segments"]

# Function to extract key screenshots
def extract_screenshots(video_path, interval=2):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * interval
    count = 0
    screenshot_paths = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            screenshot_path = f"{UPLOAD_FOLDER}/screenshot_{count}.png"
            cv2.imwrite(screenshot_path, frame)
            screenshot_paths.append(screenshot_path)
        count += 1

    cap.release()
    return screenshot_paths

# Function to extract text from UI
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(cv2.imread(image_path))

# Function to generate documentation using GPT-4
def generate_documentation(steps):
    prompt = "Generate a structured user guide with images:\n\n"
    for step in steps:
        prompt += f"Step: {step['text']}\nImage: {step['screenshot']}\n\n"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# API route to process video
@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Process video
    transcription = transcribe_audio(file_path)
    screenshots = extract_screenshots(file_path)

    # Extract text from screenshots
    steps = []
    for segment, screenshot in zip(transcription, screenshots):
        steps.append({
            "time": segment["start"],
            "text": segment["text"],
            "screenshot": screenshot
        })

    # Generate guide
    guide = generate_documentation(steps)

    return jsonify({"guide": guide, "screenshots": screenshots})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
    