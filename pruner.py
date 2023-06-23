import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

def check_module_name(name, weight, bias, norm, output_layer_name):
    
    if output_layer_name in name:
        return False
    
    if 'norm' in name and not norm:
        return False
    else:
        if 'weight' in name and weight:
            return True
        elif 'bias' in name and bias:
            return True
        else:
            return False
        
class Mask:
    def __init__(self, model, weight, bias, norm, output_layer_name):
        super().__init__()
        
        self.weight = weight
        self.bias = bias
        self.norm = norm
        self.output_layer_name = output_layer_name
        
        self.mask = self.init_mask(model)
    
    def init_mask(self, model):
        
        mask = OrderedDict()
        for name, param in model.named_parameters():
            if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                mask [name] = param.new_ones(param.size(), dtype=torch.bool)
        
        return mask
    
    def freeze_grad(self, model):
        
        for name, param in model.named_parameters():
            if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                
                mask_i = self.mask[name]
                param.grad.data = torch.where(mask_i.to(param.device), param.grad.data, torch.tensor(0, dtype=torch.float, device=param.device))
    
    def apply_mask(self, model):
        
        for name, param in model.named_parameters():
            
            if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                mask_i = self.mask[name]
                param.data = torch.where(mask_i.to(param.device), param.data, torch.tensor(0, dtype=torch.float, device=param.device))
    
    def load_state_dict(self, state_dict):
        self.mask = state_dict

    def state_dict(self):
        return self.mask
    
    def update_mask(self, model, scores, prune_scope, sparsity):
        new_mask = OrderedDict()
        
        if prune_scope == 'neuron':
            
            for name, param in model.named_parameters():
                if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                    
                    mask_i = self.mask[name]
                    pivot_param = scores[name]
                    pivot_param[~mask_i] = float('nan')
                    
                    if len(pivot_param.size()) > 1:
                        pivot_value = torch.nanquantile(pivot_param, sparsity, dim=1, keepdim=True)
                    else:
                        pivot_value = torch.nanquantile(pivot_param, sparsity, dim=0, keepdim=True)

                    pivot_mask = (scores[name] < pivot_value)
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
                    
        elif prune_scope == 'layer':
            
            for name, param in model.named_parameters():
                if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                    mask_i = self.mask[name]
                    pivot_param = scores[name][mask_i]
                    
                    pivot_value = torch.quantile(pivot_param, sparsity)
                    
                    pivot_mask = (scores[name] < pivot_value)
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
            
        elif prune_scope == 'global':
            pivot_param = []
            pivot_mask = []
            
            for name, param in model.named_parameters():
                if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                    mask_i = self.mask[name]
                    pivot_param_i = scores[name][mask_i]
                    pivot_param.append(pivot_param_i.view(-1))
                    pivot_mask.append(mask_i.view(-1))
            
            pivot_param = torch.cat(pivot_param, dim=0).data
            pivot_mask = torch.cat(pivot_mask, dim=0)
            pivot_value = np.quantile(pivot_param.data.cpu().numpy(), sparsity)
            
            for name, param in model.named_parameters():
                if check_module_name(name, self.weight, self.bias, self.norm, self.output_layer_name):
                    
                    mask_i = self.mask[name]
                    pivot_mask = (scores[name] < pivot_value)
                    new_mask[name] = torch.where(pivot_mask, False, mask_i)
                    param.data = torch.where(new_mask[name].to(param.device), param.data,
                                             torch.tensor(0, dtype=torch.float, device=param.device))
            
        else:
            raise ValueError('Not valid prune scope')
        
        self.mask = new_mask

class Pruner:
    def __init__(self, model, prune_scope, weight=True, bias=False, norm=False, output_layer_name='head'):
        super().__init__()
        
        self.weight = weight
        self.bias = bias
        self.norm = norm
        self.output_layer_name = output_layer_name
        
        self.prune_scope = prune_scope
        
        self.now_sparsity = 1.0
        self.now_iter = 0
        
        self.mask = Mask(model, self.weight, self.bias, self.norm, self.output_layer_name)
    
    def apply_mask(self, model):
        self.mask.apply_mask(model)

    def freeze_grad(self, model):
        self.mask.freeze_grad(model)
    
    def update_mask(self, model, scores, sparsity):
        self.now_iter = self.now_iter + 1
        self.now_sparsity = self.now_sparsity * (1.0 - sparsity)
        
        print('compressing to {:0.4}% ... '.format(self.now_sparsity * 100.0), end='', flush=True)
        self.mask.update_mask(model, scores, self.prune_scope, sparsity)
        print('done.')