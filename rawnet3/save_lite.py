import torch
import torchaudio
from torch.utils.mobile_optimizer import optimize_for_mobile
from model.RawNet3 import MainModel


def get_demo_wrapper():
    wrapper = torch.jit.load("rawnet3_traced.pt")    
    return wrapper

wrapper = get_demo_wrapper()
#scripted_model = torch.jit.script(wrapper)
optimized_model = optimize_for_mobile(wrapper)
optimized_model._save_for_lite_interpreter("streaming_asrv2.ptl")
print("Done _save_for_lite_interpreter")
