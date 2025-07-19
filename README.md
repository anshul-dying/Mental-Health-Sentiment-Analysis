# Mental Health Sentiment Analysis Using NLP and Deep Learning

This project implements a comprehensive mental health sentiment analysis system using NLP and deep learning techniques. It analyzes text for emotional content and autism likelihood using state-of-the-art models.

## Project Structure
```
mental-health-nlp/
├── data/                    # Dataset files
├── models/                  # Trained model weights
├── src/                     # Source code
├── static/                  # Web app static files
├── templates/               # Web app templates
├── docs/                    # Documentation and research paper
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n mental_health_env python=3.9
conda activate mental_health_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. CUDA Setup
This project requires CUDA 12.8 for GPU acceleration. Verify your setup:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

## Team Roles

1. **Data Engineer** (Weeks 1-2)
   - Data preprocessing
   - Feature extraction
   - Dataset analysis

2. **ML Engineer - Sentiment** (Weeks 3-4)
   - DistilBERT model training
   - Emotion classification
   - Model evaluation

3. **ML Engineer - Autism** (Week 5)
   - LSTM model development
   - Feature engineering
   - Model training

4. **Web Developer** (Weeks 6-7)
   - Flask app development
   - Frontend design
   - API integration

5. **Documentation & Testing** (Week 8)
   - Research paper
   - User testing
   - Deployment

## 8-Week Timeline

- **Week 1**: Data collection, preprocessing setup
- **Week 2**: Feature extraction, linguistic analysis
- **Week 3-4**: DistilBERT model training
- **Week 5**: LSTM model training
- **Week 6-7**: Flask app development and testing
- **Week 8**: Final documentation, deployment, presentation

## Usage

1. **Data Preprocessing**
```bash
python src/preprocess.py
```

2. **Model Training**
```bash
python src/train_sentiment.py
python src/train_autism.py
```

3. **Web App**
```bash
python src/app.py
```

## Requirements

- Python 3.9
- CUDA 12.8
- Conda environment
- See requirements.txt for Python dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details. 