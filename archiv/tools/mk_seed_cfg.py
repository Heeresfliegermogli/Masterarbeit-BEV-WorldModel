import sys, os, yaml
base, seed, root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open(base))
cfg["training"]["seed"]   = seed
cfg["checkpoints"]["dir"] = os.path.join(root, f"seed_{seed}")
out = f"config_nf_seed{seed}.yaml"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False, allow_unicode=True)
print(out)