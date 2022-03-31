import torch

from .hugging_utils import get_transformer
from .embed_patch.extra_embed import extra_embed

from .models import (ContextWindowModel, FullConvModel, ConvAttentionModel)
from .batchers import (ContextWindowBatcher, FullConvBatcher, ConvAttentionBatcher)

class SystemHandler:
    @classmethod
    def batcher(cls, system:str, formatting:str=None, 
                     max_len:int=None, window_len:tuple=None, C=None):
        batchers = {'window'   : ContextWindowBatcher, 
                    'whole'     : FullConvBatcher,
                    'attention' : ConvAttentionBatcher}

        batcher = batchers[system](window_len=window_len, 
                                   formatting=formatting, 
                                   max_len=max_len, 
                                   C=C)

        return batcher
    
    @classmethod
    def model(cls, transformer:str, system:str, system_args=None,
              num_labels:int=None, C:'ConvHandler'=None, formatting=None):
        """ creates the sequential classification model """

        trans_name = transformer
        trans_model = get_transformer(trans_name)

        #add extra tokens if added into tokenizer
        if len(C.tokenizer) != trans_model.config.vocab_size:
            print('extending model')
            trans_model.resize_token_embeddings(len(C.tokenizer)) 
            
        if system_args:
            trans_model = cls.patch(trans_model, trans_name, system_args)
        
        models = {'window' : ContextWindowModel, 
                  'whole'   : FullConvModel,
                  'attention' : ConvAttentionModel}
        
        model = models[system](trans_model, num_labels)
        
        print(formatting, system, trans_name)
        if (formatting=='cls') and (system=='whole') \
                               and (trans_name=='longformer'):
            print('using cls sent id')
            model.sent_id = 0

        return model
    
    @classmethod
    def patch(cls, trans_model, trans_name, system_args):
        if ('spkr_embed' in system_args) or ('utt_embed' in system_args): 
            print('using speaker embeddings')
            trans_model = extra_embed(trans_model, trans_name)

        if 'freeze-trans' in system_args:
            self.freeze_trans(transformer)
        
        return trans_model
                     
    @staticmethod
    def freeze_trans(transformer):
        for param in transformer.encoder.parameters():
            param.requires_grad = False



