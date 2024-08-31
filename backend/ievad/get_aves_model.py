from torchaudio.models import wav2vec2_model
import json
import torch
import torch.nn as nn
import logging
logger = logging.getLogger('ievad')
logger.setLevel(level=logging.DEBUG)
import yaml
with open('backend/ievad/config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

if config['embedding_model'] == 'birdaves':
    CONFIG_PATH = 'backend/ievad/models/birdaves/birdaves-bioxn-large.torchaudio.model_config.json'
    MODEL_PATH = 'backend/ievad/models/birdaves/birdaves-bioxn-large.torchaudio.pt'
else:
    CONFIG_PATH = 'backend/ievad/models/aves/aves-base-bio.torchaudio.model_config.json'
    MODEL_PATH = 'backend/ievad/models/aves/aves-base-bio.torchaudio.pt'
BATCH_SIZE = 64

class AvesTorchaudioWrapper(nn.Module):

    def __init__(self, pooling,
                 config_path=CONFIG_PATH, model_path=MODEL_PATH):

        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.pooling = pooling
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.model.eval()

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        import numpy as np
                
        for i in range(sig.shape[0]//BATCH_SIZE+1):
            batch = sig[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            out = self.model.extract_features(batch)[0][-1]
            out = getattr(torch, self.pooling)(out, dim=1)
            if i == 0:
                out_np = out.detach().numpy()
            else:
                out_np = np.append(out_np, out.detach().numpy(), axis=0)        
        return out_np
    
if __name__ == '__main__':
    torchaudio_model = AvesTorchaudioWrapper('mean', CONFIG_PATH, MODEL_PATH)
    torchaudio_model.eval()
    waveform = torch.rand((16_000))
    x = waveform.unsqueeze(0)
    a = torchaudio_model(x)
    
    