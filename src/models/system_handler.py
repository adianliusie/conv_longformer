import torch

from .hugging_utils import get_transformer
from .embed_patch.extra_embed import extra_embed

from .models import ContextWindowModel, FullConvModel
from .batchers import ContextWindowBatcher, FullConvBatcher, MaskedFullConvBatcher

class SystemHandler:
    @classmethod
    def batcher(cls, system:str, formatting:str=None, 
                     max_len:int=None, batcher_args=None, C=None):
        batchers = {'window'    : ContextWindowBatcher, 
                    'whole'     : FullConvBatcher,
                    'whole_mask': MaskedFullConvBatcher}

        batcher = batchers[system](batcher_args=batcher_args, 
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
        
        models = {'window'    : ContextWindowModel, 
                  'whole'     : FullConvModel,
                  'whole_mask': FullConvModel}

        model = models[system](trans_model, num_labels)
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

    @staticmethod
    def parallelise(model, batcher):
        model.__class__ = FullConvModel
        batcher.__class__ = FullConvBatcher
        return model, batcher
    
    @staticmethod
    def deparallelise(model, batcher):
        model.__class__   = ContextWindowModel
        batcher.__class__ = ContextWindowBatcher
        if not (hasattr(batcher, 'past') and hasattr(batcher, 'past')):
            batcher.past, batcher.fut = 1000, 1000
        return model, batcher
    



