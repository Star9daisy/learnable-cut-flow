This is the repository for the paper "[2503.22498] Learnable cut flow for high energy physics".

The _Learnable Cut Flow_ (LCF) model interprets the training of a neural network as a search for best cut combination.

The model is implemented in Keras with Jax as backend. To install the dependencies, use:

```bash
uv sync
```

The results used in the paper are available in the `results` directory. Model checkpoints are available in the `checkpoints` directory.

The jupyter notebooks are the main entry point for different experiments run in the paper. You can check them out. By default, the notebooks will use the checkpoints and exisiting results to regenerate related figures.

If you have any questions, please contact me at `star9daisy@gmail.com`.

Please cite the paper if it is useful for your research:

```bibtex
@article{Li:2025hbv,
    author = "Li, Jing and Sun, Hao",
    title = "{Learnable cut flow for high energy physics}",
    eprint = "2503.22498",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    doi = "10.1007/JHEP11(2025)047",
    journal = "JHEP",
    volume = "11",
    pages = "047",
    year = "2025"
}
```
