# unsup3d-replication
Replication of Unsup3D originally from [Unsup3D](https://github.com/elliottwu/unsup3d)
### Training Unsup3D on CelebA Dataset
1. Install dependencies from [Unsup3D](https://github.com/elliottwu/unsup3d)
* Download neural_renderer from [this link](https://github.com/adambielski/neural_renderer) instead of the original repository.
2. I trained the model with the following configurations:
* python==3.8.3
* torch==1.9.0+cu111
3. Train the model
```
python run.py --config experiments/train_celeba.yml --gpu 0 --num_workers 4
```
4. Check the results in Tensorboard
* Logs are stored in results/celeba/logs/
* From you local machine, run
```
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>
```
* On the remote machine, run
```
tensorboard --logdir results/logs --port 6006
```
