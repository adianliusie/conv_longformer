import torch
from typing import List, Tuple
from types import SimpleNamespace
import random
from transformers import AutoTokenizer
import numpy as np

from ..utils import flatten
from abc import ABCMeta

class BaseBatcher(metaclass=ABCMeta):
    """base class that creates batches for training/eval for all tasks"""

    def __init__(self, formatting:str=None, max_len:int=None, C=None, **kwargs):
        """initialises object"""
        self.device = torch.device('cpu')
        self.max_len = max_len
        self.formatting = formatting
        self.C = C
        
    def batches(self, data:List['Conversations'], 
                      bsz:int, shuffle:bool=False):
        convs = self._prep_convs(data)
        if shuffle: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        for batch in batches:
            yield self.batchify(batch)
        #batches = [self.batchify(batch) for batch in batches]       
        #return batches
    
    def _get_padded_ids(self, ids:list)->("padded ids", "padded_mask"):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def _pad_seq(self, x:list, pad_val:int=0)->list:
        """pads input so can be put in a tensor"""
        max_len = max([len(i) for i in x])
        x_pad = [i + [pad_val]*(max_len-len(i)) for i in x]
        x_pad = torch.LongTensor(x_pad).to(self.device)
        return x_pad
       
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
    
    
class ContextWindowBatcher(BaseBatcher):
    def __init__(self, window_len:tuple, max_len:int=None, 
                       formatting:str=None, C=None):
        """initialises object"""
        super().__init__(formatting, max_len, C)
        assert (len(window_len)==2)
        self.past, self.fut = window_len
    
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkr_ids, utt_ids, utt_pos, utts = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids = self._pad_seq(spkr_ids)
        utt_ids = self._pad_seq(utt_ids)
        
        #utt_pos = list(enumerate(utt_pos)) #[(0, p1), (1, p2)
        utt_pos = torch.LongTensor(utt_pos).to(self.device) #[bsz, 1]
        labels  = [utt.label for utt in utts] 
        labels  = torch.LongTensor(labels).to(self.device)
        
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                   spkr_ids=spkr_ids, utt_ids=utt_ids, utt_pos=utt_pos)
    
    def _prep_convs(self, data:List['Conversations'], return_conv=False):
        """ windowed data preparation for context (each example is an utterance)"""
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                #get past, current and future utterances within the window
                w_s, w_e = max(i-self.past, 0), i+self.fut
                cur_u  = cur_utt.ids
                past_u = [utt.ids[1:-1] for utt in conv.utts[w_s:i]]
                fut_u  = [utt.ids[1:-1] for utt in conv.utts[i+1:w_e+1]]

                #get speaker ids
                spkrs = [utt.spkr_id[0] for utt in conv.utts[w_s:w_e+1]]
                spkrs_tok = [utt.spkr_id[1] for utt in conv.utts[w_s:w_e+1]]

                #prepare the tokens to be used as a flat input
                past_u, cur_u, fut_u, utt_pos = self._format_ids(past_u, cur_u, fut_u, spkrs_tok)
                ids = flatten(past_u) + cur_u + flatten(fut_u)
                    
                #prepare other meta information useful for the task
                spkr_ids = [[s]*len(i) for s, i in zip(spkrs, past_u+[cur_u]+fut_u)]
                spkr_ids = flatten(spkr_ids)
                c = max(self.past-i, 0) #to ensure current utt has same utt_id
                utt_ids = [[k+c]*len(i) for k, i in enumerate(past_u+[cur_u]+fut_u)]
                utt_ids = flatten(utt_ids)

                ##add example to conversation examples if under max len
                if self.max_len==None or len(utt_ids)<self.max_len:
                    conv_out.append([ids, spkr_ids, utt_ids, utt_pos, cur_utt])
                else:
                    print(len(utt_ids))
                    print('filtering conversation')
            output.append(conv_out)

        return output if return_conv else flatten(output)
        
    def _format_ids(self, past:List[list], cur:list, fut:List[list], spkrs_tok):
        """depending on mode, adds sep tokens"""
               
        def add_tok_start(tok, past, cur):
            if len(past) > 0: past[0] = [tok] + past[0]
            else: cur = [tok] + cur
            return past, cur

        def add_tok_end(tok, cur, fut):
            if len(fut) > 0:  fut[-1] = fut[-1] + [tok]
            else:                 cur = cur     + [tok]
            return cur, fut

        CLS, SEP = cur[0], cur[-1]
        utt_pos = None  #position of the special token of utt
        
        if self.formatting == 'utt_sep':  # [CLS] U1 [SEP] U2 [SEP] Ui [SEP] U4 [SEP] U5 [SEP]
            cur = cur[1:-1] + [SEP]
            past = [utt + [SEP] for utt in past]
            fut  = [utt + [SEP] for utt in fut]
            past, cur = add_tok_start(CLS, past, cur)
            utt_pos = len(flatten(past + [cur])) - 1

        elif self.formatting == 'no_sep':     # U1  U2 [CLS] Ui [SEP] U4 U5
            cur = [CLS] + cur[1:-1] + [SEP]
            utt_pos = len(flatten(past))
            
        elif self.formatting == 'spkr_sep': # [CLS] [A] U1 [B] U2 [A] Ui [B] U4 [A] U5 [SEP]
            past_len = len(past)
            cur  = [spkrs_tok[past_len]] + cur[1:-1]
            past = [[s] + utt for utt, s in zip(past, spkrs_tok[:past_len])]
            fut  = [[s] + utt for utt, s in zip(fut,  spkrs_tok[past_len+1:])]
            cur, fut = add_tok_end(SEP, cur, fut)
            utt_pos = len(flatten(past)+1)              #pos of spkr tok
            past, cur = add_tok_start(CLS, past, cur)   #added after to simplify selecting above

        elif self.formatting == 'cls_wrap':          # [SEP] U1 [SEP] U2 [CLS] Ui [SEP] U4 [SEP] U5 [SEP]
            cur = [CLS] + cur[1:-1] + [SEP]
            past = [utt + [SEP] for utt in past]
            fut  = [utt + [SEP] for utt in fut]
            if past: 
                past[-1] = past[-1][:-1] #remove SEP before CLS
                past, cur = add_tok_start(SEP, past, cur)
            utt_pos = len(flatten(past))
            
        else:
            raise ValueError('invalid context formatting')
        return past, cur, fut, utt_pos

class FullConvBatcher(BaseBatcher):
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkr_ids, utt_ids, utt_pos_seq, convs = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids = self._pad_seq(spkr_ids)
        utt_ids = self._pad_seq(utt_ids)
        
        utt_pos_seq = [torch.LongTensor(utt).to(self.device) for utt in utt_pos_seq] #[bsz, 1]

        labels = [[utt.label for utt in conv] for conv in convs]
        labels = self._pad_seq(labels, pad_val=-100)
        #^keep in mind labels are wrong for seq2seq training
        
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                   spkr_ids=spkr_ids, utt_ids=utt_ids, utt_pos=utt_pos_seq)
    
    def _prep_convs(self, data:List['Conversations']):
        """ sequence classification input data preparation"""
        output = []
        for conv in data:
            #get all utterances in conv and labels
            ids = [utt.ids for utt in conv.utts]
            spkrs = [utt.spkr_id[0] for utt in conv.utts]
            spkrs_tok = [utt.spkr_id[1] for utt in conv.utts]
            ids, utt_pos_seq = self._format_ids(ids, spkrs_tok)

            #get utterance meta information
            spkr_ids = [[s]*len(i) for s, i in zip(spkrs, ids)]
            spkr_ids = flatten(spkr_ids)
            utt_ids = [[k]*len(i) for k, i in enumerate(ids)]
            utt_ids = flatten(utt_ids)
            ids = flatten(ids)
                            
            #add to data set    
            if self.max_len==None or len(utt_ids)<self.max_len:
                output.append([ids, spkr_ids, utt_ids, utt_pos_seq, conv])
                
        return output
    
    def _format_ids(self, utts, spkrs_tok):
        CLS, SEP = utts[0][0], utts[0][-1]
        utt_pos_seq = None  #position of all utt special tokens
        
        # [CLS] U1 [SEP] U2 [SEP] ... [SEP] UN [SEP] 
        if self.formatting == 'utt_sep':
            utt_ids = [utt[1:] for utt in utts]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_pos_seq = np.cumsum([len(utt) for utt in utt_ids])-1
            
        # [CLS] [A] U1 [B] U2 ... [A] UN [SEP] 
        elif self.formatting == 'spkr_sep':
            utt_ids = [[s] + utt[1:-1] for utt, s in zip(utts, spkrs_tok)]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_ids[-1] = utt_ids[-1] + [SEP]
            
        # [CLS] U1 U2 U3 ... UN [SEP] 
        elif self.formatting == 'no_sep':
            utt_ids = [utt[1:-1] for utt in utts]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_ids[-1] = utt_ids[-1] + [SEP]
            
        else:
            raise ValueError('invalid sequence formatting')
        return utt_ids, utt_pos_seq

