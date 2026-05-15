import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

class OLMoModel(nn.Module):
    def __init__(
        self,
        model_path: str,
    ):
        super(OLMoModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


if __name__ == "__main__":
    model_name_or_path = "/mnt/zzbnew/rnamodel/shenhaojie/signalDNAmodel/test-haojieshen-model-type26-cnn_type13_teacher_model_distill0.1_VQ_64k_lemon/base"

    base_model = OLMoModel(
        model_path=model_name_or_path,
    )

    print(base_model)

    




    


