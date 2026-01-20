from viewer.engine.runners.base import BaseRunner


class MetaTrainRunner(BaseRunner):
    def __init__(
        self,
        model,
        device,
        log,
        *,
        build=None,
        sample=None,
        train=None,
        eval=None,
        eval_every=0,
    ):
        super().__init__(model, device, log)
        self.build, self.sample, self.train, self.eval, self.eval_every = (
            build,
            sample,
            train,
            eval,
            eval_every,
        )
        self.k = 0

    def start(self):
        if self.build:
            self.build()
            self.log("meta-train ready")

    def step(self):
        if not (self.sample and self.train):
            return
        e, t, batch = self.sample()
        metrics = self.train(batch) or {}
        self.k += 1
        msg = f"meta k={self.k}"
        if "loss" in metrics:
            msg += f" loss={metrics['loss']:.4f}"
        self.log(msg)
        if self.eval and self.eval_every and self.k % self.eval_every == 0:
            self.log(str(self.eval()))
