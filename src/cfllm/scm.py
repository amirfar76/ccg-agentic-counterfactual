from dataclasses import dataclass

@dataclass
class ExogenousNoise:
    ollama_seed: int
    rng_run: int  # NS-3 RNG run (RngSeedManager::SetRun)
