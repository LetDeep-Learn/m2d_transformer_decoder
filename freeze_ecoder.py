import torch
import torch.nn as nn
# from encoder import CNNEncoder 
from model_ssl import CNNEncoder  # 🔁 Update with your actual encoder module
import os
from config import freeze
# 🔧 Load trained encoder
encoder = CNNEncoder()
checkpoint_path = freeze  # 🔁 Update path if needed

if os.path.exists(checkpoint_path):
    encoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    print("✅ Encoder weights loaded.")
else:
    print("❌ Checkpoint not found. Make sure training is complete.")
    exit()

# 🧊 Freeze all parameters
for param in encoder.parameters():
    param.requires_grad = False

# 🧪 Set to evaluation mode
encoder.eval()

# 🧠 Optional: move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)

# ✅ Save frozen encoder (optional)
torch.save(encoder.state_dict(), 'checkpoints/frozen_encoder.pth')
print("🧊 Encoder frozen and ready for feature extraction.")
