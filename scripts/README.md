# Scripts

Planned thin entry points only:

- `run_agent.py`: online object-centric inference.
- `train_vlm.py`: offline multimodal instruction tuning.
- `evaluate.py`: change / geometry / semantic / agent evaluation.

Business logic should remain under `src/vlm_agent/` rather than being implemented inside scripts.
