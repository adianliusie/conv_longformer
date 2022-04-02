import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

#for cross attention plots
import seaborn as sns
import matplotlib.pyplot as plt

from src.helpers import ConvHandler, Batcher, DirManager
from src.models import make_model
from src.utils import join_namespace, no_grad, Levenshtein
from .train_handler import TrainHandler
from .config import config

class EvalHandler(TrainHandler):
    """"base class for running all sequential sentence 
        evaluation and analysis on trained models"""
    
    def __init__(self, exp_name:str, hpc:bool=False):
        self.dir = DirManager.load_dir(exp_name, hpc)

    def set_up(self, args):
        #load training arguments and adding to args
        t_args = self.dir.load_args('train_args')
        args = join_namespace(args, t_args)
        self.mode = args.mode
        
        #load final model
        if not hasattr(self, 'model'):
            self.model = make_model(system=args.system, 
                                    mode=args.mode,
                                    num_labels=args.num_labels, 
                                    system_args=args.system_args)
            self.load_model()
            self.model.eval()

        #conversation processing
        self.C = ConvHandler(label_path=args.label_path, 
                             system=args.system, 
                             punct=args.punct, 
                             action=args.action, 
                             hes=args.hes, 
                             tqdm_disable=True)
        
        self.batcher = Batcher(mode=args.mode, 
                               num_labels=args.num_labels,
                               max_len=args.max_len, 
                               system_args=args.system_args)

        #get start, pad and end token for decoder
        if self.mode == 'seq2seq':
            self.decoder_start = self.model.start_idx
            self.decoder_end   = self.model.end_idx
            self.decoder_pad   = self.model.pad_idx
            
        #set to device
        self.device = args.device
        self.to(self.device)
        return args
    
    ######  Methods For Dialogue Act Classification  ##########
    
    @no_grad
    def evaluate(self, args:namedtuple):
        """ evaluating model performance with loss and accurancy"""
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare data
        eval_data = self.C.prepare_data(path=args.test_path, 
                                        lim=args.lim)
        eval_batches = self.batcher(eval_data, 
                                    bsz=args.bsz, 
                                    shuf=False)
        logger = np.zeros(3)
        for k, batch in enumerate(eval_batches, start=1):
            output = self.model_output(batch)
            
            if False:
                print(torch.argmax(output.logits.squeeze(0), -1).tolist())
                return None
            logger += [output.loss.item(), 
                       output.hits, 
                       output.num_preds]
            
        loss, acc = logger[0]/k, logger[1]/logger[2]
        print(f'loss {loss:.3f}   acc {acc:.3f}')
        return (loss, acc)
    
    #TEMP METHOD PLEASE DELETE IN FUTURE
    @no_grad
    def eval_debug(self, args:namedtuple):
        args = self.set_up(args)

        eval_data = self.C.prepare_data(path=args.test_path, 
                                        lim=args.lim)
        eval_batches = self.batcher(eval_data, 
                                    bsz=args.bsz, 
                                    shuf=False)
        logger = np.zeros(5)
        for k, batch in enumerate(eval_batches, start=1):
            output = self.model_output(batch)
            
            pred_seq = torch.argmax(output.logits.squeeze(0), -1).tolist()
            label_seq  = batch.labels[0].tolist()
            logger[:4] += Levenshtein.wer(pred_seq, label_seq)
            logger[4]  += len(label_seq)
            
            
        print(f'loss {logger[0]/k:.3f}  ',
              f'acc {logger[1]/logger[2]:.3f}')
          
        print(f"WER:{logger[0]/logger[4]:.3f}  ",
              f"replace:{logger[1]/logger[4]:.3f}  ",
              f"inserts: {logger[2]/logger[4]:.3f}  ",
              f"deletion: {logger[3]/logger[4]:.3f}")
                       
    @no_grad
    def evaluate_free(self, args:namedtuple):
        """ evaluating model in free running set up
            performance is assessed using Lev Dist."""
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare data
        eval_data = self.C.prepare_data(path=args.test_path, 
                                        lim=args.lim)
        eval_convs = self.batcher.eval_batches(eval_data)
        
        logger = np.zeros(5)
        for conv in tqdm(eval_convs):
            pred_seq   = self.model_free(conv).tolist()
            label_seq  = conv.labels.tolist()
            
            #if batch has to be squeezed as setting is seq2seq
            if len(label_seq) == 1: 
                pred_seq  = pred_seq[0][1:]
                label_seq = label_seq[0]
            
            if False:
                e, s, i, d = Levenshtein.wer(pred_seq, label_seq)
                #print(f'E:{e}, S:{s}, I:{i}, D:{d}, len:{len(label_seq)} acc:{e/len(label_seq)}')
                print(pred_seq)
                return None
                
            logger[:4] += Levenshtein.wer(pred_seq, label_seq)
            logger[4]  += len(label_seq)
            
        wer, s, i, d = [logger[i]/logger[4] for i in range(4)]
        print(f"WER: {wer:.3f}  subs: {s:.3f}  ",
              f"ins: {i  :.3f}  dels: {d:.3f}")
        return (wer, s, i, d)
              
    @no_grad
    def model_free(self, batch):
        if self.mode == 'seq2seq':
            max_len = config.debug_len if config.debug else 500
            pred = self.model.generate(
                    input_ids=batch.ids, 
                    attention_mask=batch.mask, 
                    num_beams=5,
                    bos_token_id=self.decoder_start,
                    eos_token_id=self.decoder_end,
                    pad_token_id=self.decoder_pad,
                    max_length=max_len
                   )
            
        if self.mode == 'context':
            y = self.model(input_ids=batch.ids, 
                           attention_mask=batch.mask)
            pred = torch.argmax(y, -1)
        
        return pred
        
    ######  Methods For Interpretability and Analysis   ##########
    
        
    def saliency(self, args:namedtuple, N:int=50, 
                 conv_num:int=0, utt_num:int=0):
        """ generate saliency maps for predictions """
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare conversation in interest
        conv = self.C.prepare_data(path=args.test_path)[conv_num]
        conv = self.batcher([conv], bsz=1, shuf=False)
        if self.mode == 'context': conv = conv[utt_num]
        else:                      conv = conv[0]
                
        #Get details of the max prob prediction
        y = self.model_output(conv).logits[0]
        if self.mode in ['seq2seq', 'encoder']: y = y[utt_num]
        pred_idx = torch.argmax(y).item()      
        prob = F.softmax(y, dim=-1)[pred_idx].item()
        print(self.C.label_dict[pred_idx], round(prob, 3))
        
        #get InteGrad batches (refer to paper for details)
        with torch.no_grad():
            input_embeds = self.model.get_embeds(conv.ids) #[1,L,d]
            base_embeds = torch.zeros_like(input_embeds)   #[1,L,d]
            vec_dir = (input_embeds-base_embeds)

            alphas = torch.arange(1, N+1, device=self.device)/N
            line_path = base_embeds + alphas.view(N,1,1)*vec_dir            
            batches = [line_path[i:i+args.bsz] for i in 
                       range(0, len(line_path), args.bsz)] #[N,L,d]     
                
        #Computing the line integral, 
        output = torch.zeros_like(input_embeds)
        for embed_batch in tqdm(batches):
            embed_batch.requires_grad_(True)
            
            if self.mode == 'seq2seq':
                y = self.model(inputs_embeds=embed_batch,
                               labels=conv.labels)
                y = y.logits[:,k]
                
            if self.mode == 'context':
                y = self.model(inputs_embeds=embed_batch)

            preds = F.softmax(y, dim=-1)[:,pred_idx]
            torch.sum(preds).backward()
            
            grads = torch.sum(embed_batch.grad, dim=0)
            output += grads.detach().clone()
        
        #get attribution summed for each word
        words = [self.C.tokenizer.decode(i) for i in conv.ids[0]]
        tok_attr = torch.sum((output*vec_dir).squeeze(0), dim=-1)/N
        tok_attr = tok_attr.tolist()
        
        return words, tok_attr
    
    @no_grad
    def attention(self, args, conv_num=0, free=False):
        """ get cross attention scores for a specific conversation """
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)
        
        #prepare conversation in interest
        conv = self.C.prepare_data(path=args.test_path)[conv_num]
        conv = self.batcher([conv], bsz=1, shuf=False)[0]        
        if free: conv.labels = self.model_free(conv)[:,1:] #remove [CLS]

        #forward pass
        output = self.model(input_ids=conv.ids,
                            attention_mask=conv.mask, 
                            labels=conv.labels, 
                            output_attentions=True)
        
        cross_attentions = output.cross_attentions        
        cross_attentions = [i.squeeze(0) for i in cross_attentions]
        stacked_attentions = torch.cat(cross_attentions,  dim=0)

        utt_edges = [0] + [k for k, tok in enumerate(conv.ids[0])\
                          if tok == self.C.tokenizer.eos_token_id]
        utt_attentions = self.utt_cross_attention(
                            attentions=stacked_attentions,
                            utt_edges=utt_edges)
        
        return utt_attentions.cpu().numpy()
    
    def utt_cross_attention(self, attentions, utt_edges):
        output = torch.zeros((*attentions.shape[:-1], len(utt_edges)))
        output[:, :, 0] = attentions[:, :, 0]
        for k in range(1, len(utt_edges)):
            st, end = utt_edges[k-1], utt_edges[k]
            x = torch.sum(attentions[:, :, st+1:end+1], dim=-1)
            output[:, :, k] = x
        return output
    
    @no_grad
    def pos_encodings(self, args, conv_num=0, free=False):
        
        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare conversation in interest
        conv = self.C.prepare_data(path=args.test_path)[conv_num]
        conv = self.batcher([conv], bsz=1, shuf=False)[0]        
        if free: conv.labels = self.model_free(conv)[:,1:] #remove [CLS]

        output = self.model(input_ids=conv.ids,
                            attention_mask=conv.mask, 
                            labels=conv.labels, 
                            output_hidden_states=True)
        
        hid_len = output.decoder_hidden_states[0].size(1)
        decoder_hidden = [i.squeeze(0) for i in output.decoder_hidden_states]
        dec_pos = self.model.model.decoder.embed_positions.weight[:hid_len]
        dec_pos = F.normalize(dec_pos, p=2, dim=1)
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sim = torch.matmul(dec_pos, dec_pos.T)
        sim = sim.cpu().numpy()
        print(sim[:5, :5])
        ax = sns.heatmap(sim, cbar=False, square=True)
        plt.show()
        
        for hid_vec in decoder_hidden:
            hid_vec =  F.normalize(hid_vec, p=2, dim=1)
            sim = torch.matmul(hid_vec, dec_pos.T)
            sim = sim.cpu().numpy()

            ax = sns.heatmap(sim, cbar=False, square=True)
            plt.show()
            
        #reduce size from 1024 to whatever, and show similarity for each layer
        print(output.decoder_hidden_states[0].shape)
        print(output.encoder_hidden_states[0].shape)

    @no_grad
    def position_accuracy(self, args):
        """calculates accuracy at a position level over eval data"""

        #load training arguments, model, batcher etc.
        args = self.set_up(args)

        #prepare data
        eval_data = self.C.prepare_data(path=args.test_path, 
                                        lim=args.lim)
        eval_batches = self.batcher.eval_batches(eval_data)
        
        #init positionwise accuracy tracker
        hits, counts = np.zeros(500), np.zeros(500)
        
        for k, batch in enumerate(eval_batches, start=1):
            output = self.model_output(batch)
            preds = torch.argmax(output.logits, dim=-1).squeeze(0)
            
            print(preds.shape)
            hits[:len(preds)] += (preds==batch.labels).squeeze(0)\
                                  .cpu().numpy()
            counts[:len(preds)] += 1
                
        return hits, counts
        
