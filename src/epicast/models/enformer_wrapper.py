import torch
import enformer_pytorch

class EnformerWrapper(torch.nn.Module):
    """Wrap enformer_pytorch model to match Trainer's model(sample) interface.
    forward(sample) expects a dict with key 'seq' -> Tensor [B, L, 4]
    returns predictions Tensor [B, T, C]
    """
    def __init__(self, pretrained_path, output_head='human', **kwargs):
        super().__init__()
        self.output_head = output_head
        self.enformer = enformer_pytorch.from_pretrained(pretrained_path, **kwargs)

    def forward(self, sample):
        if isinstance(sample, dict):
            seq = sample['seq']
        elif isinstance(sample, list):
            seq = sample[0]
        else:
            seq = sample
        
        out = self.enformer(seq)

        if isinstance(out, dict):
            pred = out[self.output_head]

        return pred
