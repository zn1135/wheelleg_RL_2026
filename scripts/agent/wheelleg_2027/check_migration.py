"""Check the declared environment move, old imports and retained policy loading."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path

from ct_mjlab.envs import task_registry

ROOT = Path(__file__).resolve().parents[3]
FOLDERS = ("chuanliantui_split", "chuanliantui_stand", "chuanliantui_stand_v2",
           "fudan_ppo_walk", "fudan_ppo_step", "fudan_ppo_jump")


class Normalize(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for folder in FOLDERS:
                node.value = node.value.replace(f"ct_mjlab.envs.chuanliantui.{folder}",
                                                f"ct_mjlab.envs.{folder}")
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr == "physics_cfg_factory":
                return ast.Name(id="make_env_cfg", ctx=ast.Load())
            if node.attr == "wheel_profile_factory":
                return ast.Attribute(value=ast.Name(id="wheel_physics", ctx=ast.Load()),
                                     attr="record", ctx=ast.Load())
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name == "FudanEnv":
            node.body = [statement for statement in node.body
                         if not (isinstance(statement, ast.Assign) and
                                 any(isinstance(t, ast.Name) and t.id in ("physics_cfg_factory", "wheel_profile_factory")
                                     for t in statement.targets))]
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / "artifacts/2027_wheelleg/20261126-port"
    entries = json.loads((base / "env_migration.json").read_text())
    for entry in entries:
        old_path = base / "before" / entry["old"]
        new_path = ROOT / entry["new"]
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == entry["before"]
        before = Normalize().visit(ast.parse(old_path.read_text()))
        current_text = new_path.read_text()
        if "fudan_ppo_step/" in entry["new"]:
            current_text = current_text.replace("Path(__file__).resolve().parents[6]",
                                                "Path(__file__).resolve().parents[5]")
        after = Normalize().visit(ast.parse(current_text))
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), entry
        compile(new_path.read_text(), str(new_path), "exec")
        assert not (ROOT / entry["old"]).exists(), entry["old"]
        entry["after"] = hashlib.sha256(new_path.read_bytes()).hexdigest()
    (base / "env_migration.json").write_text(json.dumps(entries, indent=2) + "\n")

    aliases = []
    for name in ("fudan_ppo_walk.fudan_ppo_walk_config", "fudan_ppo_walk.fudan_ppo_walk",
                 "fudan_ppo_step.config", "fudan_ppo_jump.config", "chuanliantui_split.env_cfg"):
        legacy = importlib.import_module("ct_mjlab.envs." + name)
        canonical = importlib.import_module("ct_mjlab.envs.chuanliantui." + name)
        assert legacy is canonical
        aliases.append(name)
    expected = ("smp", "supermimic", "fudan_ppo_walk", "fudan_ppo_step", "fudan_ppo_jump", "2027_wheelleg_walk")
    assert task_registry.names() == expected
    for name in ("chuanliantui", "chuanliantui_stand", "chuanliantui_stand_v2", "chuanliantui_getup", "chuanliantui_walk"):
        try:
            task_registry.get_cfgs(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Retired task was restored: {name}")
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    old_checkpoint = ROOT / "logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt"
    policy, saved = load_policy(old_checkpoint)
    assert next(policy.parameters()).isfinite().all()
    report = dict(passed=True, moved_files=len(entries), body_ast_equivalence=True,
                  declared_changes=["canonical imports", "two deeper repository path lookups",
                                    "overridable physics/profile factories with identical chuanliantui defaults"],
                  registered_tasks=list(expected), historical_aliases=aliases,
                  old_policy_load=True, old_policy_iterations=saved["iterations"],
                  optimizer_updates=0, physics_steps=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
