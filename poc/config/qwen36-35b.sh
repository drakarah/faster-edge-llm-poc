# Qwen3.6-35B-A3B has a discrete-GPU profile only (its experts stream from host memory); config/qwen36-35b.dgpu.sh is loaded when PROFILE=dgpu.
die "MODEL=qwen36-35b needs PROFILE=dgpu (no Orin profile: not measured there)"
