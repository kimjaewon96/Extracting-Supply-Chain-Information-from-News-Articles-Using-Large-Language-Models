from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class BertEntity(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        E1_mask = (input_ids == 30522).unsqueeze(-1)
        E2_mask = (input_ids == 30523).unsqueeze(-1)
        # input_ids = torch.where(
        #     (input_ids == 30522) | (input_ids == 30523),
        #     torch.tensor(30524).to(input_ids.device),
        #     input_ids,
        # )
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        E1_embedding = (sequence_output * E1_mask).sum(dim=1)
        E2_embedding = (sequence_output * E2_mask).sum(dim=1)
        sliced_sequence = torch.cat([E1_embedding, E2_embedding], dim=-1)

        logits = self.classifier(sliced_sequence)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
