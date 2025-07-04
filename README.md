Reproducing parts of results from NPCL: Neural Processes for Uncertainty-Aware Continual Learning (Jha et al. 2024)

The entire repo is forked from https://github.com/srvCodes/NPCL

To install dependencies, run
```
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

To test S-Tiny-ImageNet dataset under the recommended settings given by the original paper, run command line taken from [https://github.com/srvCodes/NPCL/blob/main/runner.sh](https://github.com/srvCodes/NPCL/blob/main/runner.sh)
```
python utils/main.py --dataset seq-tinyimagenet --np_type npcl --model er --kl-g 0.01 --kl-t 0.01 --kd-tr 0.1 --kd-gr 0.05 --context-batch-factor 0.125 --visualize-latent --load_best_args --seed 1  --buffer_size 500 --use_context --num_labels 5 --forward_times_train 15 --forward_times_test 15 
```

To test S-CIFAR100 dataset under the recommended settings given by the original paper, run the command line 
```
python utils/main.py --dataset seq-cifar100 --np_type npcl --model er --kl-g 0.01 --kl-t 0.05 --kd-tr 0.1 --kd-gr 0.08 --context-batch-factor 0.125 --visualize-latent --load_best_args --seed 1  --buffer_size 500 --use_context --num_labels 5 --forward_times_train 15 --forward_times_test 15 
```


