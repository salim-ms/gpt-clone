import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

from model.layers.feed_forward import MLP
from model.layers.efficient_causal_attention import EfficientCausal
from model.layers.layer_norm import LayerNormBiasEnabled

class Block(nn.Module):
    def __init__(self, embed_size, n_heads, block_size, is_bias, dropout) -> None:
        super(Block, self).__init__()    
        self.norm1 = LayerNormBiasEnabled(embed_size, bias=is_bias)
        self.norm2 = LayerNormBiasEnabled(embed_size, bias=is_bias)
        self.mlp = MLP(embed_size=embed_size, is_bias=is_bias, dropout=dropout)
        self.attn = EfficientCausal(embed_size=embed_size, n_heads=n_heads, block_size=block_size, is_bias=is_bias, dropout=dropout)
        
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layers, embed_size, n_heads, block_size, is_bias=False, dropout=0.0) -> None:
        super(GPT, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.block_size = block_size
        self.is_bias = is_bias
        self.dropout = dropout
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_size),
            wpe = nn.Embedding(block_size, embed_size),
            drop = nn.Dropout(dropout),
            layers = nn.ModuleList([Block(embed_size=embed_size, n_heads=n_heads, block_size=block_size, is_bias=is_bias, dropout=dropout) for _ in range(n_layers)]),
            ln = LayerNormBiasEnabled(embed_size, bias=is_bias)
        ))
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        # initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_parameters()/1e6,))
    
    
    def get_num_parameters(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size # make sure sequence length is not greater than configured "1024"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx) # b, t, embed_size
        tok_pos = self.transformer.wpe(pos) # b, t, embed_size
        x = self.transformer.drop(tok_emb + tok_pos)
        
        for block in self.transformer.layers:
            x = block(x)
        
        x = self.transformer.ln(x)
        
        if targets is not None:
            # we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    
    def crop_block_size(self, desired_block_size):
        # we are doing model surgery here to change pre-configured block size to a smaller one
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert desired_block_size < self.block_size
        self.block_size = desired_block_size
        
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:desired_block_size])
        # loop over the layers and assign adjust masking trill
        for block in self.transformer.layers:
            if hasattr(block.attn, "masking"):
                block.attn.masking = block.attn.masking[:, :, :desired_block_size, :desired_block_size]
        
    
    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        # enable weight decay on any weight with dimension >= 2
        # # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        
        # start with all of the candidate parameters
        param_dict = {pn:p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=betas, **extra_args)
        
        print(f"using fused AdamW: {use_fused}")
        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_parameters()
        
        L, H, Q, T = self.n_layers, self.n_heads, self.embed_size//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    
    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        
        # create a from-scratch initialized minGPT model
        model = GPT(vocab_size=config_args["vocab_size"],
            n_layers=config_args["n_layer"],
            embed_size=config_args["n_embd"],
            n_heads=config_args["n_head"],
            block_size=config_args["block_size"],
            is_bias=config_args["bias"],
            dropout=0.0)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.masking")]
        
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        

        return model