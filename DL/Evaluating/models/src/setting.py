import torch

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

data_root_path = './sampledata'

# region ablation study default
USE_SENSOR_PROCESS = True
USE_FULL_SENSOR_SEQUENCE = True #恒定
USE_LSTM_LAYER = False
USE_LEARNABLE_MAPPING = True


learning_rate = 0.00001
num_epochs = 2
top_k = 7
# endregion

ollama_url = 'http://localhost:11434/api/generate'
llm_model_name = 'llama3.1:8b'