import torch
import torch.nn as nn
import copy

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
        embeds = self.transformer.embeddings(input_ids)
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
            output[conv_num] = conv_vecs.tolist()
            
        #pad array
        pad_elem = [0]*H.size(-1)
        max_row_len = max([len(row) for row in output])
        padded_output = [row + [pad_elem]*(max_row_len-len(row))
                                              for row in output]

        return torch.FloatTensor(padded_output).to(h.device)

class ConvAttentionModel(torch.nn.Module):
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(256, num_class)
        self.small_transformer = AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.attn = nn.MultiheadAttention(embed_dim=256, kdim=768, vdim=768, num_heads=1)

    def forward(self, **kwargs):
        """for now assumes bsz size is 1"""
        
        alt_ids = kwargs.pop('alt_ids')[0]
        alt_mask = kwargs.pop('alt_mask')[0]
        
        queries = self.small_transformer(input_ids=alt_ids, 
                 attention_mask=alt_mask).last_hidden_state  #[N, L, 268] 
        queries = queries[:,0].unsqueeze(0)                  #[1, N, 268]
        
        H = self.transformer(**kwargs).last_hidden_state     #[1, L, 768] 
        
        q = torch.transpose(queries, 0, 1)     #[N, 1, 268]
        H = torch.transpose(H, 0, 1)           #[L, 1, 268]
        h = self.attn(q, H, H)                 
        h = h[0].squeeze(1)                    
        y = self.classifier(h)                 #[1, L, 43]
        return y
    
class TempSmallBERT(torch.nn.Module):
    def __init__(self, transformer, num_class):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("prajjwal1/bert-mini")
        self.classifier = nn.Linear(256, num_class)
        self.pooling = lambda h: h[:,0] #default to first vector

    def forward(self, **kwargs):
        """for now assumes bsz size is 1"""
        alt_ids = kwargs.pop('alt_ids')[0]
        alt_mask = kwargs.pop('alt_mask')[0]
        
        h   = self.transformer(input_ids=alt_ids, 
                 attention_mask=alt_mask).last_hidden_state
        h_n = self.pooling(h=h)
        y   = self.classifier(h_n)
        return y
    
class HierModel(torch.nn.Module):
    def __init__(self, transformer, num_class):
        self.transformer = transformer
        self.classifier = nn.Linear(768, num_class)
        