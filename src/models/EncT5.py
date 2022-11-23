from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Config, T5PreTrainedModel
import torch
import torch.nn as nn

from .MiddleModel import MiddleModule

class EncT5MultiRegressModel(MiddleModule):

    def __init__(self, 
        dropout=0.1,
        lr : float = 1e-5,
        mean: float = None, 
        std: float = None, 
    ):
        super().__init__("EncT5MultiRegress", lr = lr, mean = mean, std = std)

        config = T5Config.from_pretrained("t5-base")

        self.num_labels = config.num_labels
        self.config = config

        self.encoder = T5EncoderModel.from_pretrained("t5-base")

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(config.hidden_size, 6)

    def forward(self,batch):
        index, token_ids, attn_masks, y = batch

        outputs = self.encoder(
            input_ids=token_ids,
            attention_mask=attn_masks
        )

        hidden_states = outputs[0]
        pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

        pooled_output = self.dropout(pooled_output)
        ret = self.regressor(pooled_output)

        return ret

if __name__ == "__main__":
    model = T5EncoderModel.from_pretrained("t5-base",torch_dtype=torch.float16)