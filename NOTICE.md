# NOTICE

The `sotopia/` directory in this repository is a modified snapshot of the
open-source **Sotopia** framework
(https://github.com/sotopia-lab/sotopia, MIT License), taken at upstream
commit `224a371` and extended with the game-engine layer described in the
paper. All names, copyright notices, and author attributions that appear
inside `sotopia/` refer to the original Sotopia authors.

The contribution claimed by this paper is limited to:

- the **game-engine layer** that extends Sotopia's abstractions to support
  finite-state-machine games with role-conditioned visibility,
- the **21 game implementations** under `games/`,
- the **Elo tournament infrastructure** and the **SPaRTan
  play-reflect-transfer pipeline** under `experiments/`,
- and the experimental data, reflection playbooks, and analyses reported in
  the paper.

The snapshot is vendored so the repository is self-contained and reproducible. 
Our own code is released under the MIT License (see `LICENSE`).
