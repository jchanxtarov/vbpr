# vbpr
repository to implement Visual Bayesian Personalized Ranking

> title: VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback
> authors: Ruining He, Julian McAuley
> link: https://arxiv.org/abs/1510.01784

## Instllation
```bash
git clone git@github.com:jchanxtarov/vbpr.git
cd vbpr
make setup
```

## Usage Example
### Under the default setting

If you want to save log and best model, please run following command.
```
make run
```

A command to run the code without saving the log or model data (dry-run) is prepared as follow.
```
make dry-run
```

### With your own seting
```
poetry run python src/main.py {options}
```
About options, please [see here](https://github.com/jchanxtarov/vbpr#aguments).

## Aguments
To see all argments and description, please run the following command.
```bash
make args
```

## Refenrence
- https://arxiv.org/abs/1510.01784
