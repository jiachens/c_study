To adversarially train a new model run the command

```
python main.py --exp_name=<experiment_name> --model=<model_name> --num_points=1024 --k=20 --use_sgd=True --eps=<eps> --alpha=<alpha> --train_iter=<train_iter> --adversarial=True
```

To use ATTA to train run the command 

```
python main.py --exp_name=<experiment_name> --model=<model_name> --num_points=1024 --k=20 --use_sgd=True --eps=<eps> --alpha=<alpha> --train_iter=<train_iter> --adversarial=True --atta=True --atta_reset=<atta_reset>
```


To evaluate an existing model on eps in {0.025,0.05,0.075,0.1} and alpha=eps/10 run the command

```
python main.py --model=<model_name> --num_points=1024 --k=20 --eval=True --model_path=<path_to_model> --test_iter=<test_iter>         
```

If architecture modifier command line options were used for training, make sure to use them on the eval command as well.