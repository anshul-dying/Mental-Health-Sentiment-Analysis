from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os
import random
import pandas as pd

# Configure Gemini API
try:
    genai.configure(api_key="AIzaSyBsxygNtADPwsLC1n8kTapHXQsEq9z7cEU")
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Warning: Could not configure Gemini API: {str(e)}")
    model = None

app = Flask(__name__)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define EmotionClassifier
class EmotionClassifier(torch.nn.Module):
    def __init__(self, n_classes=6):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Define AutismClassifier
class AutismClassifier(torch.nn.Module):
    def __init__(self):
        super(AutismClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)
        return probs

# Initialize models
sentiment_model = None
autism_model = None

# Load models
try:
    sentiment_model = EmotionClassifier().to(device)
    sentiment_model.load_state_dict(torch.load('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/sentiment_model/best_model.pth', map_location=device))
    sentiment_model.eval()
    print("Emotion model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model files not found. Please train the models first.")
    print("To train models, run:")
    print("1. python src/train_sentiment.py")
    print("2. python src/train_autism.py")

try:
    autism_model = AutismClassifier().to(device)
    autism_model.load_state_dict(torch.load('D:/College/SY/Sem2/EDAI/Project/mental_health_Revised_model/models/autism_model/best_model.pth', map_location=device))
    autism_model.eval()
    print("Autism model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model files not found. Please train the models first.")
    print("To train models, run:")
    print("1. python src/train_sentiment.py")
    print("2. python src/train_autism.py")

def get_mental_health_suggestions(emotion_scores):
    """Get mental health suggestions based on emotion scores."""
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    if model is None:
        return (
            "Mental health suggestions are currently unavailable.\n\n"
            "**General Mental Wellness Tips**:\n"
            "- Practice mindfulness meditation for 5-10 minutes daily.\n"
            "- Maintain a consistent sleep schedule.\n"
            "- Engage in physical activity for at least 30 minutes each day."
        )
        
    suggestions = []
    has_strong_emotions = False
    
    for emotion, score in zip(emotions, emotion_scores):
        if score > 0.3:
            has_strong_emotions = True
            prompt = f"Provide 2-3 brief suggestions for managing {emotion} in a healthy way, using markdown format."
            try:
                response = model.generate_content(prompt)
                suggestions.append(f"**{emotion.capitalize()} Suggestions**:\n{response.text}")
            except Exception as e:
                print(f"Error generating suggestions for {emotion}: {str(e)}")
                fallback_suggestions = {
                    'sadness': "- **Journal**: Write down your thoughts to process emotions.\n- **Connect**: Reach out to a friend for support.\n- **Self-Care**: Engage in activities you enjoy.",
                    'joy': "- **Savor**: Reflect on what’s making you happy.\n- **Share**: Tell others about your joy.\n- **Plan**: Schedule more joyful activities.",
                    'love': "- **Express**: Share your feelings with loved ones.\n- **Bond**: Spend quality time with them.\n- **Self-Love**: Practice positive affirmations.",
                    'anger': "- **Breathe**: Use deep breathing to calm down.\n- **Pause**: Take a moment before reacting.\n- **Express**: Share feelings constructively.",
                    'fear': "- **Ground**: Focus on your breath to stay calm.\n- **Support**: Talk to someone you trust.\n- **Relax**: Try mindfulness or meditation.",
                    'surprise': "- **Reflect**: Process unexpected events.\n- **Calm**: Use deep breathing.\n- **Clarify**: Ask questions to understand."
                }
                suggestions.append(f"**{emotion.capitalize()} Suggestions**:\n{fallback_suggestions.get(emotion, 'No suggestions available.')}")
    
    if not has_strong_emotions:
        try:
            prompt = "Provide 3 brief, practical mental wellness tips in markdown format."
            response = model.generate_content(prompt)
            return f"**General Mental Wellness Tips**:\n{response.text}"
        except Exception as e:
            print(f"Error generating general suggestions: {str(e)}")
            return (
                "**General Mental Wellness Tips**:\n"
                "- Practice mindfulness meditation for 5-10 minutes daily.\n"
                "- Maintain a consistent sleep schedule.\n"
                "- Engage in physical activity for at least 30 minutes each day."
            )
    
    return "\n\n".join(suggestions)

def get_autism_analysis(text, likelihood):
    """Get autism analysis and suggestions."""
    if model is None:
        return (
            "Autism analysis is currently unavailable.\n\n"
            "**General Suggestions**:\n"
            "- **Practice regular mindfulness**: Meditate for 5-10 minutes daily.\n"
            "- **Establish healthy boundaries**: Communicate limits clearly.\n"
            "- **Engage in self-care**: Exercise, sleep, and connect socially."
        )
    
    try:
        content_check_prompt = f"""Analyze if this text contains any autism-related indicators or content: '{text}'
        Return only 'YES' if it contains clear autism-related content, or 'NO' if it doesn't."""
        
        content_check = model.generate_content(content_check_prompt)
        has_autism_content = content_check.text.strip().upper() == 'YES'
        
        if likelihood > 0.5 and not has_autism_content:
            likelihood = 0.2
            
        if has_autism_content and likelihood > 0.5:
            prompt = f"""Based on this text: '{text}', which shows potential autism indicators, provide 3 specific, actionable suggestions for support and understanding. 
            Format each suggestion with a number and make the title of each suggestion bold by surrounding it with double asterisks. Example:
            1. **Provide clear structure:** Description here.
            2. **Offer sensory accommodations:** Description here."""
        else:
            prompt = f"""The provided text doesn't offer any indicators of autism. It appears to be about {get_text_topic(text)}. 
            Therefore, providing support suggestions related to autism would be inappropriate. Instead, here are three suggestions for supporting someone in this situation:
            
            1. **Acknowledge and Validate the Feelings:** [Provide specific validation based on the content]
            2. **Encourage Connecting with Support Systems:** [Suggest relevant support systems]
            3. **Develop Coping Strategies:** [Suggest specific coping strategies]"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in autism analysis: {str(e)}")
        if likelihood > 0.5 and has_autism_content:
            return (
                "**Autism Support Suggestions**:\n"
                "1. **Provide clear and structured communication**: Offer written task lists with specific deadlines.\n"
                "2. **Offer a quiet workspace**: Provide noise-canceling headphones to reduce sensory overload.\n"
                "3. **Encourage open communication**: Create a safe space for expressing needs."
            )
        else:
            return (
                "**General Mental Health Suggestions**:\n"
                "1. **Practice regular mindfulness**: Meditate for 5-10 minutes daily.\n"
                "2. **Establish healthy boundaries**: Communicate limits clearly.\n"
                "3. **Engage in self-care**: Exercise, sleep, and connect socially."
            )

def get_text_topic(text):
    """Helper function to identify the main topic of the text."""
    if model is None:
        return "the expressed situation"
    try:
        topic_prompt = f"What is the main topic or theme of this text in 3-4 words? Text: '{text}'"
        topic_response = model.generate_content(topic_prompt)
        return topic_response.text.strip()
    except Exception as e:
        print(f"Error in get_text_topic: {str(e)}")
        return "the expressed situation"

def analyze_text(text):
    """Analyze text for emotions and autism likelihood."""
    if sentiment_model is None or autism_model is None:
        return {
            'error': 'Models not trained yet. Please train the models first.',
            'emotions': {
                'sadness': 0.0,
                'joy': 0.0,
                'love': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0
            },
            'autism_likelihood': 0.0,
            'suggestions': 'Please train the models first to get suggestions.',
            'autism_analysis': 'Please train the models first to get autism analysis.'
        }
    
    # Clean text
    cleaned_text = text.lower().strip()
    
    # Get emotion predictions
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        emotion_logits = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        emotion_probs = torch.softmax(emotion_logits, dim=1).cpu().numpy()[0]
    
    # Map emotions
    emotion_map = {
        'sadness': float(emotion_probs[0]),
        'joy': float(emotion_probs[1]),
        'love': float(emotion_probs[2]),
        'anger': float(emotion_probs[3]),
        'fear': float(emotion_probs[4]),
        'surprise': float(emotion_probs[5])
    }
    
    # Get autism likelihood
    with torch.no_grad():
        autism_probs = autism_model(input_ids=input_ids, attention_mask=attention_mask)
        autism_likelihood = float(autism_probs.cpu().numpy()[0])
        autism_likelihood = min(0.9, max(0.1, autism_likelihood * 0.6))  # Dampening factor
    
    # Get mental health suggestions
    emotion_scores = [emotion_map['sadness'], emotion_map['joy'], emotion_map['love'], 
                     emotion_map['anger'], emotion_map['fear'], emotion_map['surprise']]
    suggestions = get_mental_health_suggestions(emotion_scores)
    
    # Get autism analysis
    autism_analysis = get_autism_analysis(text, autism_likelihood)
    
    return {
        'emotions': emotion_map,
        'autism_likelihood': float(autism_likelihood),
        'suggestions': suggestions,
        'autism_analysis': autism_analysis
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    
    if len(text) < 10:
        return jsonify({
            'error': 'Text must be at least 10 characters long.'
        }), 400
    
    try:
        results = analyze_text(text)
        return jsonify(results)
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/generate-sample', methods=['GET'])
def generate_sample():
    sample_type = request.args.get('type', 'anxiety')
    
    if model is not None:
        try:
            prompts = {
                'anxiety': "Generate a short, realistic first-person paragraph (2-4 sentences) that expresses anxiety without explicitly stating 'I have anxiety'. Use natural language, include physical symptoms or thought patterns common in anxiety.",
                'depression': "Generate a short, realistic first-person paragraph (2-4 sentences) that expresses depression without explicitly stating 'I have depression'. Use natural language, focus on lack of energy, motivation, or joy.",
                'sadness': "Generate a short, realistic first-person paragraph (2-4 sentences) that expresses sadness about a specific situation. Use natural language that feels authentic.",
                'happiness': "Generate a short, realistic first-person paragraph (2-4 sentences) that expresses genuine happiness or joy about a specific achievement or situation.",
                'autism': "Generate a short, realistic first-person paragraph (2-4 sentences) from the perspective of someone with autism describing their experience with sensory sensitivity, social interaction challenges, or need for routine. Do not explicitly state 'I have autism'."
            }
            
            if sample_type in prompts:
                response = model.generate_content(prompts[sample_type])
                if response and hasattr(response, 'text'):
                    text = response.text.strip()
                    if len(text) > 300:
                        truncated = text[:300]
                        last_period = truncated.rfind('.')
                        if last_period > 150:
                            text = text[:last_period+1]
                    return jsonify({"sample": text})
        except Exception as e:
            print(f"Error using Gemini API for sample generation: {str(e)}")
    
    samples = {
        'anxiety': [
            "I can't stop worrying about my upcoming presentation. My heart races every time I think about it.",
            "Everything feels overwhelming lately. I'm constantly on edge and can't relax.",
            "I keep having these intrusive thoughts that something bad will happen if I don't check things repeatedly."
        ],
        'depression': [
            "I haven't felt like myself in weeks. Nothing brings me joy anymore.",
            "Getting out of bed feels impossible most days. I'm just so tired all the time.",
            "I used to love painting, but now I can't find the energy or motivation to pick up a brush."
        ],
        'sadness': [
            "I miss my old friends. We've all drifted apart and it hurts to think about the memories.",
            "The holidays are the hardest time for me since my grandmother passed away.",
            "I failed my exam despite studying hard, and I feel like I've let everyone down."
        ],
        'happiness': [
            "I just got accepted to my dream university! All the hard work finally paid off.",
            "Spending time with my family this weekend was exactly what I needed. I feel refreshed and loved.",
            "My therapy sessions are really helping. I'm starting to see progress and it feels amazing."
        ],
        'autism': [
            "Social situations are exhausting for me. I never know the right things to say or when to say them.",
            "Certain sounds like fire alarms or balloons popping cause me intense distress that others don't understand.",
            "I have very specific routines that help me function. When they're disrupted, I struggle to adapt."
        ]
    }
    
    if sample_type in samples:
        sample = random.choice(samples[sample_type])
        return jsonify({"sample": sample})
    else:
        return jsonify({"error": "Invalid sample type"}), 400

if __name__ == '__main__':
    app.run(debug=True)