def setup(algo, P):
    """
    Setup function for meta-learning training.
    """
    # Construct default filename string if none was provided
    if P.fname is None:
        if P.data_type == "ray":
            data = f"{P.data_type}/{P.dataset}/"
            modularization = f"cells-{P.num_submodules:02}/" + (
                "fim/" if P.fim else "mod/"
            )
            algo_str = f"algo-{P.algo}/"
            model = f"{P.nerf_variant}_dir-{P.dir_encoding}_depth-{P.num_layers}_hid-{P.dim_hidden}_ch-{P.color_hidden}/"
            bg = "no_bg/" if P.no_bg_nerf else f"bg_{P.bg_hidden}/"
            training = f"initer-{P.inner_iter:02}_samples-{P.ray_samples}/"
            optimizer = f"lr-{int(P.inner_lr * 1e3):03d}-{int(P.lr * 1e6):04d}"
            fname = data + modularization + algo_str + model + bg + training + optimizer
        else:
            raise NotImplementedError("Only ray-based data_type is implemented.")
    else:
        fname = P.fname

    # Dynamically import the appropriate train_step function
    if algo in ["fomaml", "maml", "reptile"]:
        if P.data_type == "ray":
            from .meta_train_step import train_step
            from .meta_eval_step import eval_step
    else:
        raise NotImplementedError("Only gradient-based modes are implemented.")

    # Determine whether to use today's date in logging
    today = True if P.log_date else False
    fname += f"_seed-{P.seed}"
    return train_step, eval_step, fname, today
