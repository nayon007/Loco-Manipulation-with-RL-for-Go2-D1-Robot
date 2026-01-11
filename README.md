Below is a **clean, research-grade `README.md`** tailored exactly to what you built and debugged: a **Go2 quadruped with a 6-DoF arm (Go2-D1)** trained using **Deep Whole-Body Control (Deep-WBC)** on top of **Legged Gym + RSL-RL**, including all the fixes you discovered (URDF, gains, terrain, reward shaping, etc.).

You can **copy-paste this directly** into your GitHub repository root as `README.md`.

---

# Loco-Manipulation with Reinforcement Learning for Go2-D1 Robot

This repository contains a **full reproduction and extension of the Deep Whole-Body Control (Deep-WBC) framework** for a **custom Go2 quadruped robot equipped with a 6-DoF manipulator (Go2-D1)**.
The system enables **simultaneous locomotion and manipulation** using **end-to-end reinforcement learning** in **NVIDIA Isaac Gym**.

The implementation builds upon **Legged Gym** and **RSL-RL**, and includes substantial modifications to support:

* A custom URDF with arm integration
* Correct whole-body dynamics and Jacobians
* Stable reset and terrain spawning
* Proper arm/leg torque control
* Curriculum-based training for loco-manipulation

---

## 📌 Key Features

* **Whole-Body RL Control**
  End-to-end PPO policy controlling **18 actions** (12 leg + 6 arm joints)

* **Custom Go2-D1 Robot Model**

  * Go2 quadruped base
  * 6-DoF arm + 2-DoF gripper
  * Fully validated URDF and meshes

* **Deep-WBC Architecture**

  * Privileged information encoder
  * History encoder
  * Separate leg and arm control heads

* **Stable Training Pipeline**

  * Deterministic reset
  * Correct PD gain assignment
  * Name-based DOF indexing (no hard-coded slices)
  * Positive-reward curriculum
  * Proper termination handling

* **Terrain-Aware Simulation**

  * Perlin heightfields
  * Correct terrain transforms
  * Robust environment origin sampling

---

## 🧠 Based On

This work is a **reproduction and extension** of:

**Mark et al., “Learning Whole-Body Control for Legged Manipulation”**
GitHub: [https://github.com/MarkFzp/Deep-Whole-Body-Control](https://github.com/MarkFzp/Deep-Whole-Body-Control)

Key upstream components:

* **Legged Gym**
* **RSL-RL**
* **NVIDIA Isaac Gym**

All original licenses are preserved.

---

## 🏗 Repository Structure

```
Deep-WBC/
├── legged_gym/
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── base/              # Base legged robot logic
│   │   │   ├── go2d1/             # Go2-D1 task + environment
│   │   │   │   ├── go2d1.py
│   │   │   │   └── go2d1_config.py
│   │   ├── scripts/
│   │   │   ├── train.py
│   │   │   └── play.py
│   │   └── utils/
│   └── resources/
│       └── robots/
│           └── go2d1/
│               ├── meshes/
│               └── urdf/go2d1.urdf
├── rsl_rl/                         # PPO implementation
├── go2d1/                          # Standalone URDF + meshes
├── README.md
```

---

## 🤖 Robot Model (Go2-D1)

* **DOFs:** 20

  * 12 leg joints
  * 6 arm joints
  * 2 gripper joints

* **Rigid bodies:** 27

* **URDF:** `resources/robots/go2d1/urdf/go2d1.urdf`

Important design decisions:

* `collapse_fixed_joints = True`
* `base` is used instead of `trunk` (trunk is collapsed)
* Arm links: `Link1` … `Link7_1`, `Link7_2`
* Arm joints: `Joint1` … `Joint7_1`, `Joint7_2`

---

## ⚙️ Installation

### 1️⃣ System Requirements

* Ubuntu 20.04 / 22.04
* NVIDIA GPU with CUDA
* Python 3.8
* NVIDIA Isaac Gym (Preview 4)

### 2️⃣ Create Environment

```bash
conda create -n dwbc python=3.8
conda activate dwbc
```

### 3️⃣ Install Dependencies

```bash
pip install torch==1.13.1+cu117
pip install numpy wandb scipy
```

### 4️⃣ Install Legged Gym & RSL-RL

```bash
pip install -e rsl_rl
pip install -e legged_gym
```

Make sure `ISAAC_GYM_PATH` is correctly set.

---

## 🚀 Training

Train the Go2-D1 loco-manipulation policy:

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task go2d1 \
  --headless \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 0 \
  --run_name go2d1_wbc_fixed
```

### ✔️ What to Expect

* Episodes stabilize (>400 steps)
* Mean reward becomes positive
* Action noise decays
* Robot stands, walks, and manipulates without falling

---

## ▶️ Playing a Trained Policy

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task go2d1 \
  --load_run go2d1_wbc_fixed \
  --checkpoint 39999
```

Recommended evaluation overrides in `play.py`:

```python
env_cfg.env.num_envs = 1
env_cfg.domain_rand.randomize_friction = False
env_cfg.noise.add_noise = False
```

---

## 🧩 Key Fixes Implemented (Important)

This repository includes **critical fixes** beyond the original Deep-WBC codebase:

### ✔️ URDF Consistency

* Removed invalid `trunk` references
* Matched code to actual body names (`base`, `Link*`)

### ✔️ PD Gain Assignment

* Eliminated substring collisions (`joint` vs `Joint`)
* Explicit gain mapping per joint type

### ✔️ Energy Penalty Indexing

* Replaced hard-coded DOF slices with name-based indexing

### ✔️ Terrain Spawning Bug

* Fixed invalid origin bounds for small terrains
* Corrected terrain transforms
* Ensured robots spawn above the ground

### ✔️ Reward Shaping

* Enabled positive reward clipping
* Added termination penalty
* Rebalanced torque and energy penalties

### ✔️ Action Safety

* Reduced `clip_actions` from 100 → 1.0

These fixes are **required** for stable learning on custom robots.

---

## 📊 Final Training Results

At the end of training:

* **Mean episode length:** ~421 steps
* **Mean reward:** positive
* **Dones:** ~0.00
* **Policy:** stable, deterministic
* **Behavior:** standing, walking, arm reaching

This confirms correct whole-body coordination.

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@misc{go2d1_deepwbc,
  title={Loco-Manipulation with Reinforcement Learning for Go2-D1 Robot},
  author={Hafiz Nayon},
  year={2026},
  note={Based on Deep Whole-Body Control}
}
```

Also cite the original Deep-WBC paper by Mark et al.

---

## ⚠️ Notes

* Old checkpoints trained with incorrect URDF/gains **are not compatible**
* Always retrain after changing DOF order, URDF, or reward structure
* Isaac Gym preview builds are required

---

## 🙏 Acknowledgements

* **Legged Robotics**
* **ETH Zürich**
* **NVIDIA Isaac Gym Team**
* **Mark Fzp et al. (Deep-WBC)**

