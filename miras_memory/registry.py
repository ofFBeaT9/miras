"""
Project Registry — Manages isolated MIRAS memory instances per project.
Each project gets its own memory config, its own storage, its own lifecycle.
"""

import json
import time
from pathlib import Path
from typing import Optional
from .engine import ProjectMemory, MIRASConfig, PRESETS


class ProjectRegistry:
    """
    Central registry for all projects and their MIRAS memory instances.
    Think of this as the "database" of projects.
    """

    def __init__(self, base_dir: str = "./memory_store"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ProjectMemory] = {}
        self._registry_path = self.base_dir / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {"projects": {}}

    def _save_registry(self):
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def create_project(self, project_id: str, name: str = "",
                       description: str = "",
                       preset: str = "titans_default",
                       config: Optional[dict] = None) -> ProjectMemory:
        """
        Create a new project with its own MIRAS memory.

        Args:
            project_id: unique identifier
            name: human-readable name
            description: what this project is about
            preset: one of "titans_default", "yaad_robust", "moneta_strict",
                    "memora_stable", "lightweight"
            config: override specific MIRAS settings (merged with preset)
        """
        if project_id in self._registry["projects"]:
            raise ValueError(f"Project '{project_id}' already exists")

        # Build config from preset + overrides
        miras_config = PRESETS.get(preset, PRESETS["titans_default"])
        if config:
            config_dict = miras_config.to_dict()
            config_dict.update(config)
            miras_config = MIRASConfig.from_dict(config_dict)

        # Register
        self._registry["projects"][project_id] = {
            "name": name or project_id,
            "description": description,
            "preset": preset,
            "created_at": time.time(),
            "config": miras_config.to_dict(),
        }
        self._save_registry()

        # Create and cache memory instance
        mem = ProjectMemory(
            project_id=project_id,
            config=miras_config,
            storage_dir=str(self.base_dir / project_id),
        )
        self._cache[project_id] = mem
        return mem

    def get_project(self, project_id: str) -> ProjectMemory:
        """Get or load a project's memory"""
        if project_id not in self._registry["projects"]:
            raise KeyError(f"Project '{project_id}' not found")

        if project_id not in self._cache:
            reg = self._registry["projects"][project_id]
            config = MIRASConfig.from_dict(reg["config"])
            self._cache[project_id] = ProjectMemory(
                project_id=project_id,
                config=config,
                storage_dir=str(self.base_dir / project_id),
            )
        return self._cache[project_id]

    def list_projects(self) -> list[dict]:
        """List all projects with summary stats"""
        projects = []
        for pid, info in self._registry["projects"].items():
            entry = {
                "project_id": pid,
                "name": info["name"],
                "description": info["description"],
                "preset": info["preset"],
                "created_at": info["created_at"],
            }
            # Add live stats if cached
            if pid in self._cache:
                entry["stats"] = self._cache[pid].stats()
            projects.append(entry)
        return projects

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and its memory"""
        if project_id not in self._registry["projects"]:
            return False

        # Remove from cache
        self._cache.pop(project_id, None)

        # Remove from registry
        del self._registry["projects"][project_id]
        self._save_registry()

        # Remove storage
        storage = self.base_dir / project_id
        if storage.exists():
            import shutil
            shutil.rmtree(storage)

        return True

    def update_config(self, project_id: str, config_updates: dict) -> MIRASConfig:
        """Update a project's MIRAS configuration"""
        if project_id not in self._registry["projects"]:
            raise KeyError(f"Project '{project_id}' not found")

        reg = self._registry["projects"][project_id]
        current = reg["config"]
        current.update(config_updates)
        new_config = MIRASConfig.from_dict(current)

        reg["config"] = new_config.to_dict()
        self._save_registry()

        # Reload memory with new config
        self._cache.pop(project_id, None)
        return new_config
