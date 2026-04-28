import matplotlib.pyplot as plt
import numpy as np
import time
import os
from src.config import ExperimentConfig
from src.data_manager import DataManager
import shutil
import yaml
from itertools import product


LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]
LINECOLORS = ["red", "blue", "green", "orange"]

    

class SaveResults():
    
    def __init__(self, config: ExperimentConfig, results: dict, data_manager: DataManager):
        
        self.config = config
        self.results = results
        self.data_manager = data_manager
        self.dir_path = None
        
        
    def _setup_dir(self):
        
        dir_name = f"{self.config.run_name}_{str(time.time())[:8]}"
        self.dir_path = os.path.join(os.getcwd(), dir_name)

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        else:
            print(f"A directory with the name {dir_name} already exists")
            print(f"Overriding the existing directory")
            shutil.rmtree(self.dir_path)
            os.makedirs(self.dir_path)

        # Save config as YAML
        config_path = os.path.join(self.dir_path, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config.__dict__, f)

        print("Directory made")
        
        
    # TODO: this is slightly overcomplicated, take a look at this later
    def _set_plot_label(self):
        # Check if all runs have the same K
        same_K = len(set(run.K for run in self.config.runs)) == 1
        # Check if all runs use the same training method
        same_training = len(set(run.dynamics_mode for run in self.config.runs)) == 1
        # Check if any run uses online training
        is_online = any(run.dynamics_mode == "online" for run in self.config.runs)
        same_tied = True
        same_scale = True
        if is_online:
            same_tied = len(set(getattr(run, "tied", False) for run in self.config.runs if run.dynamics_mode == "online")) == 1
            same_scale = len(set(getattr(run, "scale_online", False) for run in self.config.runs if run.dynamics_mode == "online")) == 1

        labels = []
        for run in self.config.runs:
            label_parts = []
            label_parts.append(run.activation_function)
            if run.dynamics_mode == "analytical_integral":
                label_parts.append(run.ode_method)
            if not same_K:
                label_parts.append(f"K={run.K}")
            if not same_training:
                label_parts.append(f"Method={run.dynamics_mode}")
            if is_online and run.dynamics_mode == "online":
                if not same_tied:
                    label_parts.append(f"Tied={getattr(run, 'tied', False)}")
                if not same_scale:
                    label_parts.append(f"ScaleOnline={getattr(run, 'scale_online', False)}")
            labels.append(", ".join(label_parts) if label_parts else "Run")

        # Make plot title
        title_parts = []
        if not same_K:
            title_parts.append("K= (" + ", ".join(str(run.K) for run in self.config.runs) + ")")
        else:
            title_parts.append(f"K={self.config.runs[0].K}")
        if not same_training:
            title_parts.append("Method= (" + ", ".join(str(run.dynamics_mode) for run in self.config.runs) + ")")
        else:
            title_parts.append(f"Method={self.config.runs[0].dynamics_mode}")
        if is_online:
            if not same_tied:
                title_parts.append("Tied= (" + ", ".join(str(getattr(run, "tied", False)) for run in self.config.runs if run.dynamics_mode == "online") + ")")
            if not same_scale:
                title_parts.append("ScaleOnline= (" + ", ".join(str(getattr(run, "scale_online", False)) for run in self.config.runs if run.dynamics_mode == "online") + ")")
        plot_title = "; ".join(title_parts)

        

        return labels, plot_title

    def _plot_order_parameters(self):
        STYLE = {
            "analytical_integral": "solid",
            "hermite_integral": "dashdot",
            "online": "dashed",
        }

        def _colors():
            return plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

        def _elems(K: int):
            if K == 2:
                return list(product(range(K), range(K))), "entries"
            return [(i, i) for i in range(K)], "diagonals"

        def _canon_bool(x):
            return False if x is None else bool(x)

        def _base_key(run_cfg):
            dm = run_cfg.dm_config
            # Canonicalize fields that may be None in some runs due to YAML omissions
            return (
                dm.dataset,
                getattr(dm, "rank", None),
                getattr(dm, "batch_size", None),
                run_cfg.D,
                run_cfg.K,
                run_cfg.alpha,
                run_cfg.learning_rate,
                run_cfg.activation_function,
                _canon_bool(getattr(run_cfg, "tied", False)),
                _canon_bool(getattr(run_cfg, "scale_online", False)),
                _canon_bool(getattr(run_cfg, "bias", False)),
            )

        # ---- 1) joint_dynamics: overlay analytical (solid) vs SGD (dashed), same color ----
        for run_idx, result in enumerate(self.results):
            run_cfg = result["run_config"]
            if run_cfg.dynamics_mode != "joint_dynamics":
                continue

            logs = result["logs"]
            alpha = float(run_cfg.alpha)

            for name in ["T", "T0", "Q1", "R1"]:
                a_vals = logs.get(name, [])
                s_vals = logs.get(f"{name}_sgd", [])
                if not a_vals:
                    continue

                arr_a = np.asarray(a_vals)
                arr_s = np.asarray(s_vals) if s_vals else None
                if arr_a.ndim != 3:
                    continue

                alpha_a = np.linspace(0, alpha, num=len(arr_a))
                alpha_s = np.linspace(0, alpha, num=len(arr_s)) if arr_s is not None else None

                plt.figure()
                _, K, _ = arr_a.shape
                elems, kind = _elems(K)
                cols = _colors()

                for e_idx, (i, j) in enumerate(elems):
                    c = cols[e_idx % len(cols)] if cols else None

                    plt.plot(
                        alpha_a,
                        arr_a[:, i, j],
                        color=c,
                        linestyle="solid",
                        label=f"{name}[{i},{j}] analytical",
                    )

                    if arr_s is not None and arr_s.ndim == 3:
                        plt.plot(
                            alpha_s,
                            arr_s[:, i, j],
                            color=c,
                            linestyle="dashed",
                            label=f"{name}[{i},{j}] SGD",
                        )

                plt.xscale("log")
                plt.legend()
                plt.xlabel("α")
                plt.ylabel(f"{name} {kind}")
                plt.title(f"{name} for joint_dynamics (run {run_idx})")
                plt.tight_layout()
                plt.savefig(f"{self.dir_path}/{name}_joint_dynamics_run{run_idx}.png")
                plt.close()

        # ---- 2) analytical/hermite/online: group counterparts, overlay with same colors ----
        groups = {}
        for run_idx, result in enumerate(self.results):
            run_cfg = result["run_config"]
            mode = run_cfg.dynamics_mode
            if mode not in STYLE:
                continue

            logs = result.get("logs", {})
            if not any(k in logs for k in ("T", "T0", "Q1", "R1")):
                continue

            groups.setdefault(_base_key(run_cfg), []).append((run_idx, result))

        mode_priority = {"analytical_integral": 0, "hermite_integral": 1, "online": 2}

        for group_idx, (gkey, items) in enumerate(groups.items()):
            cfg0 = items[0][1]["run_config"]
            dm0 = cfg0.dm_config
            title_prefix = f"{dm0.dataset}; K={cfg0.K}; lr={cfg0.learning_rate}; act={cfg0.activation_function}"

            for name in ["T", "T0", "Q1", "R1"]:
                series = []
                for run_idx, result in items:
                    run_cfg = result["run_config"]
                    mode = run_cfg.dynamics_mode
                    vals = result["logs"].get(name, [])
                    if not vals:
                        continue

                    arr = np.asarray(vals)
                    if arr.ndim != 3:
                        continue

                    series.append((mode, float(run_cfg.alpha), arr))

                if not series:
                    continue

                series.sort(key=lambda x: mode_priority.get(x[0], 999))

                plt.figure()
                _, K, _ = series[0][2].shape
                elems, kind = _elems(K)
                cols = _colors()

                for e_idx, (i, j) in enumerate(elems):
                    c = cols[e_idx % len(cols)] if cols else None
                    for mode, alpha, arr in series:
                        alpha_vals = np.linspace(0, alpha, num=len(arr))
                        pretty = {
                            "analytical_integral": "Analytical",
                            "hermite_integral": "Hermite",
                            "online": "Online",
                        }.get(mode, mode)

                        plt.plot(
                            alpha_vals,
                            arr[:, i, j],
                            color=c,
                            linestyle=STYLE.get(mode, "solid"),
                            label=f"{name}[{i},{j}] {pretty}",
                        )

                plt.xscale("log")
                plt.legend()
                plt.xlabel("α")
                plt.ylabel(f"{name} {kind}")
                plt.title(f"{title_prefix} | {name}")
                plt.tight_layout()
                plt.savefig(f"{self.dir_path}/{name}_group{group_idx}.png")
                plt.close()


    
    
    def _get_pca_errors(self):
        pca_errors = self.data_manager.pca_errors()
    
        for error in pca_errors:
            plt.axhline(
                y=error,
                color="grey",
                linewidth=1.0,
                alpha=0.6,
                zorder=1,
            )
    
    def _plot_bias(self):
        for idx, result in enumerate(self.results):
            logs = result["logs"]
            initial_bias = logs.get("initial_bias")
            final_bias = logs.get("final_bias")

            print(f"hello {initial_bias}, {final_bias}")
            
            if initial_bias is not None and final_bias is not None:
                initial_bias = np.asarray(initial_bias)
                final_bias = np.asarray(final_bias)

                plt.figure()
                K = initial_bias.shape[0]
                indices = np.arange(K)
                plt.bar(
                    indices - 0.2,
                    np.abs(initial_bias),
                    width=0.4,
                    label="Initial Bias Magnitude",
                    color="blue",
                    alpha=0.7
                )
                plt.bar(
                    indices + 0.2,
                    np.abs(final_bias),
                    width=0.4,
                    label="Final Bias Magnitude",
                    color="red",
                    alpha=0.7
                )
                plt.xlabel("K")
                plt.ylabel("Bias Magnitude")
                plt.title(f"Bias Magnitude (run {idx})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{self.dir_path}/bias_magnitude_run{idx}.png")
                plt.close()
        

    def _plot_results(self):
        
        labels, plot_title = self._set_plot_label()
        
        self._get_pca_errors()
        for (idx, result) in enumerate(self.results):
            if result["run_config"].dynamics_mode == "joint_dynamics":
                analytical_error = result["logs"].get("analytical_generalization_error", [])
                sgd_error = result["logs"].get("sgd_generalization_error", [])
                alpha = result["run_config"].alpha

                if analytical_error:
                    alpha_vals_analytical = np.linspace(0, alpha, num=len(analytical_error))
                    plt.plot(
                        alpha_vals_analytical,
                        analytical_error,
                        color=LINECOLORS[idx % len(LINECOLORS)],
                        linestyle="solid",
                        label=f"{labels[idx]} Analytical",
                    )
                if sgd_error:
                    alpha_vals_sgd = np.linspace(0, alpha, num=len(sgd_error))
                    plt.plot(
                        alpha_vals_sgd,
                        sgd_error,
                        color=LINECOLORS[idx % len(LINECOLORS)],
                        linestyle="dashed",
                        label=f"{labels[idx]} SGD",
                    )
                
            else:
                generalization_error = result["logs"]["generalization_error"]
                alpha = (result["run_config"]).alpha
                D = result["run_config"].D
                alpha_vals = np.linspace(0, alpha, num=len(generalization_error))
                
                plt.plot(
                    alpha_vals[:len(generalization_error)],
                    generalization_error,
                    color=LINECOLORS[idx % len(LINECOLORS)],
                    linestyle=LINESTYLES[idx % len(LINESTYLES)],
                    label=labels[idx],
                )
        
        
        plt.legend(title="Runs")
        plt.title(plot_title)
        plt.xlabel("α")
        plt.ylabel("Generalization Error")
        
        plt.savefig(f"{self.dir_path}/{self.config.run_name}.png")
        
        plt.xscale("log")
        plt.xlim(0, None)
        
        plt.savefig(f"{self.dir_path}/{self.config.run_name}_log.png")
        plt.close()
    
    
    def plot(self):
        
        self._setup_dir()
        self._plot_results()
        self._plot_order_parameters()
        self._plot_bias()