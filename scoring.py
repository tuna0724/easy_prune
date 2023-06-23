from collections import OrderedDict
from pruner import check_module_name

def magnitude(model, pr):
    scores = OrderedDict()
    
    for name, param in model.named_parameters():
            
        if check_module_name(name, pr.weight, pr.bias, pr.norm, pr.output_layer_name):
            scores[name] = param.data.abs().detach().clone()
        
    return scores