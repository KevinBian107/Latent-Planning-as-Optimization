# Planning is Latent Search
This repository is designed to tackle **continual learning** problems with **long horizon planning**. We try to leverages probabilistic inference along with meta-learning to build offline learning algorithms capable of learning multipl tasks and creating its own diverse strategies, achieving the so-called "learning to learn".

- [Setup environment](docs/setup.md)
- [Data Flow](docs/data.md)
- [Structure of code](docs/pipeline.md)

Run our code from the entry point with:

```bash
python mpi/train.py --task training --train_type mixed
```

Or running the legacy code:

```bash
python mpi/agent/legacy/lpt/lpt_train_maze2d.py
```

## Acknowledgements
- The decision transformer model implementation advices code from the [official repository of DT](https://github.com/kzl/decision-transformer) .
- The latent plan transformer model implementation advices code from the [official repository of LPT](https://github.com/mingluzhao/Latent-Plan-Transformer) .
