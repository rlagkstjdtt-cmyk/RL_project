# LunarLander DQN Project

OpenAI Gymnasium의 **LunarLander-v3** 환경에서  
DQN 계열 알고리즘과 보상 설계, 구성요소(타깃 네트워크, 리플레이 버퍼, ε-decay)를 실험한 프로젝트입니다.

모든 코드는 하나의 Colab 노트북(`강화학습개론_final_project.ipynb`) 안에 들어 있고,  
일반화 단계에서 학습한 최종 모델(`AutoBest_Combined.pth`)과  
발표용 PPT(`LunarLander_Project_Report.pptx`)를 함께 제공합니다.

---

## Files

- `강화학습개론_final_project.ipynb`  
  - 전체 실험 코드가 들어 있는 Colab 노트북입니다.  
  - 내용:
    - 기본 DQN / Double DQN / Dueling DQN 구현
    - 리워드 설계 실험 (Base / FuelSaving / SafeLanding 등)
    - 구성요소(ablation) 실험  
      (타깃 업데이트 방식, 리플레이 버퍼 크기, ε-decay 속도)
    - 일반화 실험 및 최종 모델 학습 (**AutoBest_Combined**)
    - 학습 곡선, 비교 그래프, 플레이 시각화 코드

- `AutoBest_Combined.pth`  
  - **일반화 단계에서 학습된 최종 에이전트(AutoBest_Combined)** 의 가중치 파일입니다.  
  - 노트북에서 정의한 `DQNAgent`의 `q_net` / `target_q_net` 가중치와  
    최종 설정(config) 등이 저장되어 있습니다.

- `LunarLander_Project_Report.pptx`  
  - 프로젝트 발표용 PPT 보고서입니다.  
  - 프로젝트 주제, 설계 내용, 구현 방법, 실험 결과를 슬라이드로 정리했습니다.

---

## Environment

이 프로젝트는 Google Colab 환경을 기준으로 작성되었습니다.

필요 주요 라이브러리 :

import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import Image, display

Colab에서는 `main.ipynb` 상단의 셀에서 `pip install`로 필요한 패키지를 설치하도록 되어 있습니다.

로컬에서 실행할 경우 예시:

```bash
pip install -q swig
pip install -q "gymnasium[box2d]"
pip install -q torch matplotlib imageio


## How to Run (Colab 기준)

1. 이 레포지토리를 ZIP으로 다운로드하거나, GitHub에서 직접 `강화학습개론_final_project.ipynb`를 엽니다.
2. Google Colab에서 `강화학습개론_final_project.ipynb`를 열어 줍니다.
   - Colab 상단 메뉴에서 `런타임 → 런타임 유형 변경`에서 GPU 선택 
3. 노트북 상단부터 셀을 순서대로 실행합니다.
   - import / 환경 설정
   - `ReplayBuffer`, `QNetwork`, `DQNAgent` 정의
   - `train_agent`, `evaluate_agent`, `run_experiments` 정의
   - 알고리즘 비교, 리워드 설계, 구성요소 실험
   - 일반화(AutoBest_Combined 학습 및 평가)
4. 실행 시간이 오래 걸리므로, 디버깅 시에는 노트북 안에 있는 주석에 따라
   - 에피소드 수(`num_episodes`)
   - seed 개수
   등을 줄여서 빠르게 확인한 뒤, 최종 실험에서만 충분한 에피소드로 돌립니다.

## Pretrained Model (AutoBest_Combined.pth)

일반화 단계에서 학습한 최종 에이전트는  
`AutoBest_Combined.pth` 파일로 저장되어 있습니다.

### Colab에서 학습 없이 바로 불러서 평가하기

`main.ipynb`에서 **모든 클래스/함수 정의 셀**(ReplayBuffer, QNetwork, DQNAgent, train_agent 등)을 먼저 실행한 뒤,  
아래와 같은 셀을 추가해서 실행하면 **저장된 모델을 불러와 바로 플레이**할 수 있습니다.

```python
import torch
import gymnasium as gym

ENV_ID = "LunarLander-v3"

# 환경 생성
env = gym.make(ENV_ID)
obs, info = env.reset(seed=0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 학습 때 사용한 것과 같은 구조/하이퍼파라미터로 DQNAgent 생성
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
    gamma=0.99,
    lr=1e-3,
    buffer_size=100_000,
    batch_size=64,
    min_replay_size=5_000,
    target_update_freq=1_000,
    eps_start=0.0,   # 평가용이므로 탐험 X
    eps_end=0.0,
    eps_decay_steps=1,
    use_double_dqn=True,    # 일반화 단계에서 사용한 알고리즘 설정에 맞게
    use_dueling=True,       # (예: Double DQN + Dueling 조합이면 True)
    use_target_net=True,
    use_soft_update=True,   # Soft update 사용했다면 True
    tau=0.005,              # 사용한 값에 맞게
)

# 저장된 가중치 로드
ckpt = torch.load("AutoBest_Combined.pth", map_location="cpu")
agent.q_net.load_state_dict(ckpt["q_net"])
agent.target_q_net.load_state_dict(ckpt["target_q_net"])
agent.q_net.eval()
agent.target_q_net.eval()

# 한 에피소드 플레이
state, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    action = agent.select_action(state, eval_mode=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state
    env.render()

env.close()
print("episode return:", total_reward)
