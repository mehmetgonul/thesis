# Online Learning in Shallow Autoencoders

Thesis experiments comparing SGD training with ODE predictions for nonlinear autoencoders.

Experiment settings are in `configs/`, implementation in `src/`, and plotting scripts in `plot/`.

We used Python 3. Set up the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run one experiment or all configurations for an activation:

```bash
python -m src.train_autoencoder configs/erf/spiked_wishart_3_k=3.yaml
bash scripts/run_erf.sh
```

Choose the activation's script in `scripts/`. YAML options `sgd` and `ode_sol` select the run modes. Running the experiment scripts automatically creates an `artifacts/` folder and saves the experiment outputs there.

Generate figures from saved `artifacts/` data by running these commands from the repository root:

```bash
python plot/plot_spikes3.py
python plot/plot_spike_counts.py
python plot/plot_paper_generalization.py
python plot/plot_paper_order_parameters.py
```

Edit settings at the top of each script. Figures are saved in `plot/output/`; paper scripts export PNG and PDF.
