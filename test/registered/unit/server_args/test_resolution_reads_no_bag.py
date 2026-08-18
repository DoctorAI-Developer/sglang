"""Resolution does not read the config bags, because they do not exist yet.

The bags are projected from what resolution decides, so anything the pipeline
calls has to read the resolving state instead — `resolved_view(server_args)`,
or the view a handler already holds. A bag read reached from resolution raises
`config namespace ... not published`, and only on the branch that reaches it:
the diffusion-LM page-size pass needed one model family, the Marlin LoRA
validation needed one MoE runner backend. Both were written, merged into a
branch, and stayed green for everything except the configuration that triggers
them.

`test_publish_precedes_bag_reads.py` is the same worry from the other side, but
it walks the *process entries* — it cannot see a helper the pipeline calls, and
neither of the two above appeared in it.

The walk starts from three places: the symbols the pipeline imports, the
passes it runs by value (`run_post_process_pass(sa, fn)` names the callable at
the call site, and `@register_post_process` marks the rest), and the override
providers the registry calls. From there it follows calls in-module, one hop
out, and matches an accessor whether it is spelled bare or through an object.

What this still cannot see: a callable that reaches the pipeline through a
variable rather than by name, and a bag read behind an import the walk does not
follow. It is a ratchet, not a proof.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"


def _accessor_names():
    """Every bag accessor `runtime_context` exports, read from the module.

    Listing them by hand is how this went stale once already: the list had
    eighteen names while the module exported twenty-five, so a resolution-time
    `get_flags().x` or `get_resources().y` would have walked straight past.
    """
    tree = ast.parse((_SRT / "runtime_context.py").read_text(encoding="utf-8-sig"))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (node.name.startswith("get_") or node.name.startswith("configured_"))
    }
    # The context object itself is not a bag: it exists before anything is
    # published, and `declare_late_resolution` calls it deliberately to find
    # out whether the record it was handed has been published yet.
    return frozenset(names - {"get_context"})


_BAG_ACCESSORS = _accessor_names()

# `get_device` is also the name of the device-string utility and of the
# platform method it calls (`current_platform.get_device(device_id)`), so only
# the bare spelling is the accessor. The collision is not hypothetical: one
# module importing both under that name is what shadowed the accessor in the
# expert-distribution recorder.
_ATTRIBUTE_SPELLED = _BAG_ACCESSORS - {"get_device"}

# The pipeline itself and the mechanism it publishes through: `runtime_context`
# defines the accessors, and `arg_groups` is the pipeline's own code.
_OWN = ("server_args.py", "runtime_context.py")


def _module_of(name):
    """`sglang.srt.a.b` -> the file, if it is one of ours."""
    if not name or not name.startswith("sglang.srt."):
        return None
    rel = name[len("sglang.srt.") :].replace(".", "/")
    for candidate in (_SRT / f"{rel}.py", _SRT / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imported_symbols(paths):
    """{module file: {symbol names imported from it}} across the given sources."""
    out = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _module_of(node.module)
            if target is None or target.name in _OWN:
                continue
            out.setdefault(target, set()).update(alias.name for alias in node.names)
    return out


def _registered_entries():
    """Entries the import map cannot reach: passes and override providers.

    A pass arrives at the pipeline as a value, and the registry calls its
    providers by dictionary lookup. Both run during resolution, so a bag read
    inside one raises exactly like a bag read in a handler -- and neither is
    named by an import the walk can follow.
    """
    registrars = (
        "register_post_process",
        "register_model_override",
        "register_model_override_predicate",
    )
    entries = set()
    for path in sorted(_SRT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    root = (
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    if isinstance(root, ast.Name) and root.id in registrars:
                        entries.add((path, node.name))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_post_process_pass"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Name)
            ):
                entries.add((path, node.args[1].id))
    return entries


def _reaches_a_bag(path, entry):
    """Does `entry` in `path` reach a bag accessor, following calls in-module?"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    seen = set()

    def walk(name):
        if name in seen or name not in functions:
            return None
        seen.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                # `rc.get_exec()`, `self.get_schedule()`: the same accessor
                # reached through a module alias or an object.
                if node.func.attr in _ATTRIBUTE_SPELLED:
                    return node.lineno
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id in _BAG_ACCESSORS:
                return node.lineno
            found = walk(node.func.id)
            if found is not None:
                return found
        return None

    return walk(entry)


class TestResolutionReadsNoBag(CustomTestCase):
    def test_the_accessor_set_is_derived_and_whole(self):
        """A shrunken accessor set would make every other check pass quietly."""
        self.assertGreaterEqual(
            len(_BAG_ACCESSORS),
            20,
            f"only {len(_BAG_ACCESSORS)} accessors were derived from "
            "runtime_context; the derivation broke",
        )
        # The namespaces resolution most plausibly reaches for, spelled out so
        # a rename that drops one from the module is a failure here rather
        # than a silently narrower walk.
        for name in ("get_exec", "get_flags", "get_parallel", "get_resources"):
            self.assertIn(name, _BAG_ACCESSORS)

    def test_the_walk_finds_something_to_walk(self):
        """A collapsed import map would make the pin vacuous."""
        imported = _imported_symbols(
            [_SRT / "server_args.py", _SRT / "arg_groups" / "overrides.py"]
        )
        self.assertGreater(
            len(imported),
            20,
            f"the pipeline only imports from {len(imported)} of our modules; "
            "the scan broke",
        )

    def test_the_registered_entries_are_found(self):
        """The passes and providers are the half the import map cannot see."""
        entries = _registered_entries()
        self.assertGreater(
            len(entries),
            30,
            f"only {len(entries)} passes and providers were found; the scan broke",
        )

    def test_nothing_the_pipeline_calls_reads_a_bag(self):
        imported = _imported_symbols(
            [_SRT / "server_args.py", _SRT / "arg_groups" / "overrides.py"]
        )
        reachable = {
            (path, symbol) for path, symbols in imported.items() for symbol in symbols
        } | _registered_entries()
        found = []
        for path, symbol in sorted(reachable):
            line = _reaches_a_bag(path, symbol)
            if line is not None:
                found.append(
                    f"{path.relative_to(_SRT)}:{line} reached from "
                    f"{symbol}(), which resolution calls"
                )
        self.assertEqual(
            found,
            [],
            "resolution reaches a config-bag read, which raises on whichever "
            "branch gets there first; read the resolving state instead:\n  "
            + "\n  ".join(found),
        )


if __name__ == "__main__":
    unittest.main()
