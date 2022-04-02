import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm
from abc import ABCMeta
import itertools

from .eval_handler import BaseLoader
from ..helpers import (ConvHandler, DirManager, SSHelper)
from ..models import SystemHandler
from ..utils import (no_grad, toggle_grad, make_optimizer, 
                     make_scheduler)
    
class IntegradHandler(BaseLoader):
    def saliency(self, args:namedtuple, N:int=50, conv_num:int=0, 
                       utt_num:int=0, quiet=False):
        """ generate saliency maps for parallel model """

        #method only works for deparallelised system
        self.deparallelise()
        
        #prepare conversation in interest
        eval_data = self.C.prepare_data(path=args.eval_path, lim=args.lim)
        conv = eval_data[conv_num]
        convs = self.batcher(data=[conv], bsz=1, shuffle=False)
        print(utt_num)
        conv_b = next(itertools.islice(convs, utt_num, None)) #sellect specific utt

        #Get details of the max prob prediction
        y = self.model_output(conv_b).logits[0]
        pred_idx = torch.argmax(y).item()      
        prob = F.softmax(y, dim=-1)[pred_idx].item()
        if not quiet: print(self.C.label_dict[pred_idx], round(prob, 3))
        
        #get InteGrad batches (refer to paper for details)
        with torch.no_grad():
            input_embeds = self.model.get_embeds(conv_b.ids) #[1,L,d]
            base_embeds = torch.zeros_like(input_embeds)     #[1,L,d]
            vec_dir = (input_embeds-base_embeds)

            alphas = torch.arange(1, N+1, device=self.device)/N
            line_path = base_embeds + alphas.view(N,1,1)*vec_dir            
            batches = [line_path[i:i+args.bsz] for i in 
                       range(0, len(line_path), args.bsz)] #[N,L,d]     

        #repeat position idx for batch
        utt_pos = conv_b.utt_pos.repeat(args.bsz)

        #Computing the line integral, 
        output = torch.zeros_like(input_embeds)
        for embed_batch in tqdm(batches, disable=quiet):            
            embed_batch.requires_grad_(True)
            y = self.model({'inputs_embeds':embed_batch}, utt_pos=utt_pos)
            preds = F.softmax(y, dim=-1)[:, pred_idx]
            torch.sum(preds).backward()

            grads = torch.sum(embed_batch.grad, dim=0)
            output += grads.detach().clone()
        
        #get attribution summed for each word
        words = [self.C.tokenizer.decode(i) for i in conv_b.ids[0]]
        tok_attr = torch.sum((output*vec_dir).squeeze(0), dim=-1)/N
        tok_attr = tok_attr.tolist()
        return words, tok_attr

        
