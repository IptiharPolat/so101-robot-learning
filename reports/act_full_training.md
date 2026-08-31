# ACT 30K cloud training report

Date: 2026-08-25 (Asia/Shanghai)
Status: **PASS; PRIVATE HF READBACK PASS**
Real-robot evaluation: **not started**

## Run identity

- Dataset: `iptihar/so101_pink_cyan_sequence_act_v1`
- Dataset revision: `v3.0` / `cd6183684f54f942d1fa29f0b7f6390983ec2250`
- Policy: ACT trained from scratch
- LeRobot: 0.4.4, clean checkout `0f392484458cb5ebca0310c0c4c47390a31c80ed`
- PyTorch: 2.7.1+cu126
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Batch size: 8
- Seed: 20260824
- Steps: 30,000
- Sample exposure: 240,000 samples, approximately 7.23 dataset epochs
- Output: `outputs/train/act_two_cube_30k`
- W&B run: `9nhd04g6`

The run started at 15:53:53 and logged `End of training` at 16:41:28, a
wall-clock interval of approximately 47 minutes 35 seconds including startup
and checkpoint saves. At the checked CNY 1.88/hour instance price, this is
approximately CNY 1.49 of GPU runtime, excluding any later idle instance time.

## Training QC

| Check | Result |
|---|---|
| Process exit code | 0 |
| Logged metric windows | 150 |
| Loss | 7.007 first window; 0.109 final window; 0.109 minimum |
| Numeric validation | All parsed loss/gradient/LR/timing values finite |
| Checkpoints | 10K, 20K, and 30K present; `last -> 030000` |
| 30K CPU reload | PASS; 51,597,190 total/trainable parameters |
| Peak GPU memory | 4,857 MiB of 24,564 MiB |
| Peak GPU utilization | 94% |
| Peak GPU power | 341.01 W |
| Fatal errors | No OOM, Python traceback, NaN, or Inf |
| W&B | `finished`, summary step 30,000, loss 0.108752 |
| W&B checkpoint artifacts | None for the formal run |

The three checkpoint directories occupy approximately 1.8 GB together. The
persistent disk had approximately 48 GB free after training. The 10K and 20K
checkpoints remain on the instance for later comparison, but only the 30K
checkpoint was published.

## Hugging Face publication and readback

- Private model repo: `iptihar/act_so101_pink_cyan_sequence_v1`
- Published revision: `d5d6fa1afa56b928808091306b2edcc7d01c200b`
- Uploaded checkpoint: 30,000 steps only
- Weight SHA-256: `dc2900eed6e179c143f817cf051efe41843f82d87b5ecd418e2fca8fef558d01`
- Included: ACT config/weights, training config, model card, policy
  preprocessor, policy postprocessor, and normalization state files

The fixed Hub revision was downloaded into a fresh temporary cache. Its weight
SHA-256 exactly matched the local 30K checkpoint, and the ACT policy, training
config, four-step preprocessor, and two-step postprocessor all reloaded from
the private Hub repository. The readback confirmed the correct training
dataset and all 51,597,190 policy parameters.

## Interpretation and next gate

This is a successfully trained fixed-order ACT baseline, not language
understanding evidence. Falling training loss and checkpoint reload prove
training integrity but do not prove real-robot task success. The next and only
recommended action is a low-speed, staged real-robot evaluation beginning with
one guarded trial. Formal seen/unseen evaluation must use the predefined rubric
and remain separately authorized.
