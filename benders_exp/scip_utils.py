from __future__ import annotations

from pyscipopt import Model


def configure_scip(model: Model, *, time_limit: float | None = None, threads: int = 1) -> None:
    model.hideOutput(True)
    model.setIntParam("display/verblevel", 0)
    model.setIntParam("parallel/maxnthreads", max(1, int(threads)))
    if time_limit is not None and time_limit > 0:
        model.setRealParam("limits/time", float(time_limit))
