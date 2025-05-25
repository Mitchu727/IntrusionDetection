# IntrusionDetection

## Virtual Environment Setup
### Creation
```bash
python -m venv intrusion_detection
```
### Activation
```bash
source intrusion_detection/bin/activate
```

### Deactivation
```bash
source deactivate
```

## Dependencies
### Installation
```bash
pip install -r requirements.txt
``` 
### Save
```bash
pip freeze > requirements.txt
```

## Run experiments
```bash
python main.py --algorithm <algorithm> # kNN or PSO+kNN
```