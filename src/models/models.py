import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence


from transformers import AutoModel 

class ContextWindowModel(torch.nn.Module):
    """wrapper where a classification head is added to trans"""
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)

    def forward(self, trans_args, utt_pos=None):
        H   = self.transformer(**trans_args).last_hidden_state #[bsz, L, 768]
        if utt_pos is None:  
            h = H[:,0]                            #[bsz, 768]
        else:
            utt_pos = utt_pos.unsqueeze(-1)         #[bsz, 1]
            utt_pos = utt_pos.repeat(1,H.size(-1))  #[bsz, 768]
            utt_pos = utt_pos.unsqueeze(1)          #[bsz, 1, 768]
            h = H.gather(1, utt_pos).squeeze(1)     #[bsz, 768]
            
        y = self.classifier(h)                  
        return y
    
    def get_embeds(self, input_ids):
        """ gets encoder embeddings for given ids"""
        embeds = self.transformer.embeddings.word_embeddings(input_ids)
        return embeds


class FullConvModel(torch.nn.Module):
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)
        
    def forward(self, trans_args, utt_pos):
        H = self.transformer(**trans_args).last_hidden_state    #[bsz, L, 768] 
        h_sents = self.get_sent_vectors(H, utt_pos)             #[bsz, N, 768] 
        y = self.classifier(h_sents)                            #[bsz, N, C]
        return y
    
    def get_sent_vectors(self, H:torch.Tensor, utt_pos_seq:'List[list]'):
        "only selects vectors where the input id is sent_id"
 
        output = [None for _ in range(len(utt_pos_seq))]
        for conv_num, utt_pos in enumerate(utt_pos_seq):
            h = H[conv_num]                         #[L, 768]
            utt_pos = utt_pos.unsqueeze(-1)         #[N, 1]
            utt_pos = utt_pos.repeat(1, H.size(-1)) #[N,768]
            conv_vecs = h.gather(0, utt_pos)        #[N,768]
            output[conv_num] = conv_vecs
            
        #pad array
        output = pad_sequence(output, batch_first=True, padding_value=0.0)
        return output

    def get_embeds(self, input_ids):
        """ gets encoder embeddings for given ids"""
        embeds = self.transformer.embeddings(input_ids)
        return embeds
    
        