Reproducing parts of results from NPCL: Neural Processes for Uncertainty-Aware Continual Learning (Jha et al. 2024)

The entire repo is forked from https://github.com/srvCodes/NPCL

To install dependencies, run
```
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

Test S-Tiny-ImageNet dataset under the recommended settings given by the original paper, run command line in _runner.sh_ from the original repo
```
python utils/main.py --model er --visualize-latent --dataset seq-tinyimagenet --load_best_args --seed 1  --buffer_size 500 --use_context --num_labels 5 --np_type npcl --forward_times_train 15 --forward_times_test 15 --kl-g 0.01 --kl-t 0.01 --kd-tr 0.1 --kd-gr 0.05 --context-batch-factor 0.125
```

Test S-CIFAR100 dataset under the recommended settings given by the original paper, modify the command line given in _runner.sh_ from the original repo
```
python utils/main.py --model er --visualize-latent --dataset seq-cifar100 --load_best_args --seed 1  --buffer_size 500 --use_context --num_labels 5 --np_type npcl --forward_times_train 15 --forward_times_test 15 --kl-g 0.01 --kl-t 0.05 --kd-tr 0.1 --kd-gr 0.08 --context-batch-factor 0.125
```

