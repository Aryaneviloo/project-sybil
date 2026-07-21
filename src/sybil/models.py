
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("sybil")


class SybilModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = config.resolved_device()
        logger.info("Loading models on device=%s", self.device)

        # Draft and target must share a tokenizer family for token-level
        # comparison to be meaningful. 
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading draft model (Oracle): %s", config.draft_model)
        self.oracle = AutoModelForCausalLM.from_pretrained(config.draft_model).to(self.device)
        self.oracle.eval()

        logger.info("Loading target model (Sovereign): %s", config.target_model)
        self.sovereign = AutoModelForCausalLM.from_pretrained(config.target_model).to(self.device)
        self.sovereign.eval()

        logger.info("Models loaded.")
