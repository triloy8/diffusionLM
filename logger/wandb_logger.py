from __future__ import annotations

from typing import Dict, Any, Optional

from .base import Logger


class WandbLogger(Logger):
    def __init__(self, entity: Optional[str] = None, project: Optional[str] = None, name: Optional[str] = None):
        self._entity = entity
        self._project = project
        self._name = name
        self._run = None

    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        # Import lazily to keep core free of wandb dependency
        import wandb  # type: ignore

        self._run = wandb.init(entity=self._entity, project=self._project, name=self._name, config=config)
        return {"run_name": self._run.name}

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        import wandb  # type: ignore

        wandb.log(data, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: Optional[str] = None) -> None:
        try:
            import wandb  # type: ignore

            art = wandb.Artifact(name or path, type=type_ or "artifact")
            art.add_file(path)
            wandb.log_artifact(art)
        except Exception:
            # Stay resilient: logging artifacts is optional.
            pass

    def finish(self) -> None:
        if self._run is not None:
            import wandb  # type: ignore

            wandb.finish()
            self._run = None

