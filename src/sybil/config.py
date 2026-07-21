
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SybilConfig:
    draft_model: str = "gpt2"
    target_model: str = "gpt2-medium"


    num_speculative_tokens: int = 5  


    prompt: str = "The fundamental nature of power is"
    max_new_tokens: int = 50
    do_sample: bool = False      
    temperature: float = 1.0
    top_p: float = 1.0

 
    num_trials: int = 5          
    num_warmup: int = 2          #its not one lucky run just to avoid it some warmup
    seed: int = 42

    # Device
    device: Optional[str] = None  

    def resolved_device(self) -> str:  #using if cuda is available 
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
