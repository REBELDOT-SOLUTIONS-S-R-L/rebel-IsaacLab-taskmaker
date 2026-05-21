# IL Task: LegoG1

Auto-generated IsaacLab IL extension for the **IL-LEGO-G1-v0** task.

## Installation

```bash
cd lego_g1
pip install -e source/lego_g1
```

## Usage

```bash
# Run with teleoperation (using the included script)
./isaaclab.sh -p lego_g1/scripts/teleop.py --task IL-LEGO-G1-v0

# Or run with random actions
./isaaclab.sh -p lego_g1/scripts/random_agent.py --task IL-LEGO-G1-v0
```

If the `scripts/` directory was not generated (no `--isaaclab-path` was provided),
you can still use IsaacLab scripts directly by adding this import to the script:

```python
import lego_g1  # noqa: F401
```

Place it after `import isaaclab_tasks` in any IsaacLab script.

## Project Structure

```
source/lego_g1/lego_g1/
├── assets/          # USD scene, object, and robot files
├── base_il_env/     # Base IL environment (shared across tasks)
└── tasks/           # Task-specific configurations and MDP functions
```

## Customization

- Edit `tasks/manager_based/lego_g1/lego_g1_cfg.py` for task configuration
- Add custom MDP functions in `tasks/manager_based/lego_g1/mdp/`
- Place USD files in the `assets/` subdirectories
