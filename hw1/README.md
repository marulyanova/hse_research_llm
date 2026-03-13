REINFORCEMENT LEARNING, результаты в файле hw1/exp_block1_report.pdf

BEHAVIRAL CLONING, результаты в файле hw1/exp_block2_report.pdf

### Запуск

1) Эксперимент на vanilla policy gradient, чтобы подобрать параметры для MLP

```sh
bash run_mlp_params_exps.py
```

2) Эксперимент на разные loss + регуляризацию энтропии с параметром

```sh
bash run_methods_exps.py
```

3) Train Behavioral Cloning

```sh
python train_bc.py
```

3) Evaluate Behavioral Cloning

```sh
python eval_bc.py
```