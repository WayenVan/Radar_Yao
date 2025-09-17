export PYTHONPATH=./src:$PYTHONPATH

accelerate launch --mixed_precision fp16 -m radar.commands.train
