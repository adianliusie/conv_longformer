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
    def saliency(self, args:namedtuple, N:int=100, conv_num:int=0, 
                       utt_num:int=0, quiet=False):
        """ generate saliency maps for parallel model """

        #prepare conversation in interest
        self.model.eval()
        eval_data = self.C.prepare_data(args.eval_path, args.lim, quiet=True)
        conv = eval_data[conv_num]
        convs = self.batcher(data=[conv], bsz=1, shuffle=False)
        conv_b = next(itertools.islice(convs, utt_num, None)) #sellect specific utt

        #Get details of the max prob prediction
        y = self.model_output(conv_b).logits[0]
        pred_idx = torch.argmax(y).item()      
        prob = F.softmax(y, dim=-1)[pred_idx].item()
        
        pred_class = self.C.label_dict[pred_idx]
        true_class = self.C.label_dict[conv_b.labels.item()]
        if not quiet: print(f'pred: {pred_class} ({round(prob, 3)})    ',
                            f'true: {true_class}')
        
        # InteGrad embedding info (refer to paper for details)
        with torch.no_grad():
            input_embeds = self.model.get_embeds(conv_b.ids) #[1,L,d]
            base_embeds = torch.zeros_like(input_embeds)     #[1,L,d]
            vec_dir = (input_embeds-base_embeds)
        
        def make_embeds(alpha):
            alpha_shape = (1, input_embeds.size(1), 1)
            alpha_vec = torch.ones(alpha_shape, device='cuda')
            embeds = base_embeds + alpha_vec*vec_dir
            return alpha_vec.view(-1), embeds
        
        #repeat position idx for batch
        utt_pos = conv_b.utt_pos.repeat(args.bsz)

        #Computing the line integral, 
        output = torch.zeros(input_embeds.size(1)).cpu()
        for alpha in tqdm(range(1,N+1), disable=quiet):  
            alpha_vec, embed = make_embeds(alpha/N)
            alpha_vec.requires_grad_(True)
            y = self.model({'inputs_embeds':embed}, utt_pos=utt_pos)
            preds = F.softmax(y, dim=-1)[:, pred_idx]
            torch.sum(preds).backward()
            print(alpha_vec.grad.shape)
            output += alpha_vec.grad.detach().cpu().clone()
        
        #get attribution summed for each word
        words = [self.C.tokenizer.decode(i) for i in conv_b.ids[0]]
        tok_attr = torch.sum((output*vec_dir).squeeze(0), dim=-1)/N
        tok_attr = tok_attr.tolist()
        return words, tok_attr, prob, pred_class, true_class

    def utt_scores(self, words, word_scores):
        words = [i.replace('<s>', '[CLS]') for i in words]
        words = [i.replace('</s>', '[SEP]') for i in words]
        s = [0] + [k for k, word in enumerate(words) if word == '[SEP]']
        utt_scores = [sum(word_scores[s[i]:s[i+1]]) for i in range(len(s)-1)]
        return utt_scores

    def conv_integrad(self, args, conv_num, N:int=100):
        conv_data = self.C.prepare_data(path=args.eval_path, lim=args.lim, quiet=True)
        conv = conv_data[conv_num]
        output = []
        for utt_num in tqdm(range(len(conv.utts))):
            words, word_scores, prob, pred_class, true_class \
                = self.saliency(args, N, conv_num, utt_num, quiet=False)
            utt_scores = self.utt_scores(words, word_scores)
            utt = {'words':words, 'word_scores':word_scores, 'prob':prob,
                   'utt_scores':utt_scores[1:], 'cls_score':utt_scores[0],
                   'pred_class':pred_class, 'true_class':true_class}
            output.append(utt)
        return output
                