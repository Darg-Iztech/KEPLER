# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from collections import OrderedDict
from .hub_interface import RobertaHubInterface


@register_model('roberta')
class RobertaModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        self.ke_heads = nn.ModuleDict()
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--bert', action='store_true', 
                            help='use bert')
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def KEscore(self, src_tokens, relations, features_only=True, return_all_hiddens=False, ke_head_name=None, **kwargs):
        heads, tails, nHeads, nTails, heads_r, tails_r, relation_desc = src_tokens
        size = heads.size(0)
        head_embs, _ = self.decoder(heads, features_only, return_all_hiddens, **kwargs)
        tail_embs, _ = self.decoder(tails, features_only, return_all_hiddens, **kwargs)
        nHead_embs, _ = self.decoder(nHeads, features_only, return_all_hiddens, **kwargs)
        nTail_embs, _ = self.decoder(nTails, features_only, return_all_hiddens, **kwargs)
        head_embs_r, _ = self.decoder(heads_r, features_only, return_all_hiddens, **kwargs)
        tail_embs_r, _ = self.decoder(tails_r, features_only, return_all_hiddens, **kwargs)
        if relation_desc is not None:
            relation_desc_emb, _ = self.decoder(relation_desc, features_only, return_all_hiddens, **kwargs)
        else:
            relation_desc_emb = None
        pScores, nScores = self.ke_heads[ke_head_name](head_embs, tail_embs, nHead_embs, nTail_embs, head_embs_r, tail_embs_r, relations, relation_desc_emb = relation_desc_emb)
        return pScores, nScores, size

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim if 'input_dim' not in kwargs else kwargs['input_dim'],
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )
    
    def register_ke_head(self, name, num_relations=None, **kwargs):
        """Register a knowledge embedding head."""
        if name in self.ke_heads:
            prev_num_relations = self.ke_heads[name].out_proj.out_features #??
            if num_classes != prev_num_classes:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '.format(
                        name, num_relations, prev_num_relations
                    )
                )
        self.ke_heads[name] = RobertaKnowledgeEmbeddingHead(
            #self.args.encoder_embed_dim,
            self.args,
        )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v
        
        #load ke head
        current_head_names = [] if not hasattr(self, 'ke_heads') else \
            self.ke_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'ke_heads.'):
                continue

            head_name = k[len(prefix + 'ke_heads.'):].split('.')[0]
            num_relations = state_dict[prefix + 'ke_heads.' + head_name + '.relation_emb.weight'].size(0)
            
            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_ke_head(head_name, num_relations)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_relations != self.ke_heads[head_name].relation_emb.weight.size(0)
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'ke_heads'):
            cur_state = self.ke_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'ke_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'ke_heads.' + k)
                    state_dict[prefix + 'ke_heads.' + k] = v



class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaKnowledgeEmbeddingHead(nn.Module):
    """Head for knowledge embedding pretraining tasks."""

    def __init__(self, args, gamma=0, nrelation=0):
        super().__init__()
        if gamma == 0:
            gamma = args.gamma
        if nrelation == 0:
            nrelation = args.nrelation

        self.args = args
        self.nrelation = nrelation
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad = False
        )
        self.hidden_dim = args.encoder_embed_dim
        self.eps = 2.0
        self.emb_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.eps) / self.hidden_dim]),
            requires_grad = False
        )
        self.relation_emb = nn.Embedding(nrelation, args.encoder_embed_dim)
        nn.init.uniform_(
            tensor = self.relation_emb.weight,
            a=-self.emb_range.item(),
            b=self.emb_range.item()
        )

        model_func = {
            'TransE': self.TransE,
            #'DistMult': self.DistMult,
            #'ComplEx': self.ComplEx,
            #'RotatE': self.RotatE,
            #'pRotatE': self.pRotatE
        }
        self.score_func = model_func[args.ke_model]

    def TransE(self, head, relation, tail):
        #print(head.size())
        #print(relation.size())
        #print(tail.size())
        score = (head + relation) -tail
        score = self.gamma.item() - torch.norm(score, p=2, dim=2)
        #print(score)
        return score

    def forward(self, heads, tails, nHeads, nTails, heads_r, tails_r, relations, relation_desc_emb=None, **kwargs):
        heads = heads[:, 0, :].unsqueeze(1)
        tails = tails[:, 0, :].unsqueeze(1)
        heads_r = heads_r[:, 0, :].unsqueeze(1)
        tails_r = tails_r[:, 0, :].unsqueeze(1)

        #nHeads = nHeads[:, 0, :].view(heads.size(0), self.args.negative_sample_size, -1)
        #nTails = nTails[:, 0, :].view(tails.size(0), self.args.negative_sample_size, -1)
        nHeads = nHeads[:, 0, :].view(heads.size(0), -1, self.args.encoder_embed_dim)
        nTails = nTails[:, 0, :].view(tails.size(0), -1, self.args.encoder_embed_dim)
        
        if relation_desc_emb is not None:
            relations = relation_desc_emb[:, 0, :].unsqueeze(1)
        else:
            relations = self.relation_emb(relations).unsqueeze(1)
        
        heads = heads.type(torch.cuda.FloatTensor)
        tails = tails.type(torch.cuda.FloatTensor)
        nHeads = nHeads.type(torch.cuda.FloatTensor)
        nTails = nTails.type(torch.cuda.FloatTensor)
        heads_r = heads_r.type(torch.cuda.FloatTensor)
        tails_r = tails_r.type(torch.cuda.FloatTensor)

        relations = relations.type(torch.cuda.FloatTensor)

        pScores = (self.score_func(heads_r, relations, tails) + self.score_func(heads, relations, tails_r)) / 2.0
        nHScores = self.score_func(nHeads, relations, tails_r)
        nTScores = self.score_func(heads_r, relations, nTails)

        USE_NHEADS = False
        if USE_NHEADS:
            nScores = torch.cat((nHScores, nTScores), dim=1) #check the shape
        else:
            nScores = nTScores
        #print("Pscore",pScores)
        #print("Nscore",nScores)
        return pScores, nScores

class RobertaEncoder(FairseqDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            bert=args.bert if 'bert' in args else False
        )
        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens, last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, **unused):
        return self.lm_head(features)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        if hasattr(self.args,'KEdata'):
            return (512,2147483647)
            #OrderedDict([
            #            ('MLM',(self.args.max_positions,self.args.max_positions)),
            #            ('KE',(self.args.max_positions,self.args.max_positions)),
            #        ])
        else:
            return self.args.max_positions


@register_model_architecture('roberta', 'roberta')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('roberta', 'roberta_base')
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture('roberta', 'roberta_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)
