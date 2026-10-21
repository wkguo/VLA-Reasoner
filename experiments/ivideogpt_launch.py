"""Runs iVideoGPT's training scripts from its own directory on a dataset exported by export_world_model_data.py."""

import argparse
import os
import runpy
import sys

PATH_ARGS = {"--dataset_path", "--output_dir", "--pretrained_model_name_or_path", "--pretrained_transformer_path",
             "--resume_from_checkpoint", "--i3d_path"}


def _absolutize(argv):
    out, i = [], 0
    while i < len(argv):
        arg = argv[i]
        key, eq, val = arg.partition("=")
        if key in PATH_ARGS and eq:
            out.append(f"{key}={os.path.abspath(val)}")
        elif arg in PATH_ARGS and i + 1 < len(argv):
            out += [arg, os.path.abspath(argv[i + 1])]
            i += 1
        else:
            out.append(arg)
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ivideogpt_root", required=True)
    parser.add_argument("--script", required=True, choices=["train_tokenizer.py", "train_gpt.py"])
    parser.add_argument("--dataset_name", default="vla_reasoner")
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    rest = _absolutize(rest)

    root = os.path.abspath(args.ivideogpt_root)
    sys.path.insert(0, root)
    os.chdir(root)
    from ivideogpt.data.dataset_mixes import DATASET_NAMED_MIXES

    DATASET_NAMED_MIXES.setdefault(args.dataset_name, [(args.dataset_name, 1.0)])
    script = os.path.join(root, args.script)
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
