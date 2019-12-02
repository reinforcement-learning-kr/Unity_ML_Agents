# RLKorea Unity ML-agents 튜토리얼 프로젝트

이 레포지토리는 [Reinforcement Learning Korea](<https://www.facebook.com/groups/ReinforcementLearningKR/?ref=bookmarks>)의 [Unity ML-agents](<https://unity3d.com/kr/machine-learning>) 튜토리얼 프로젝트를 위한 레포입니다. 이 레포는 유니티 ML-Agents([Github](<https://github.com/Unity-Technologies/ml-agents>))로 만든 간단한 환경들을 제공합니다. 또한 제공된 환경들에서 에이전트를 학습할 수 있는 심층강화학습 알고리즘을 제공합니다. 



## 버전 정보

- **Unity**: 2019.1
- **Python**: 3.6
- **Tensorflow**: 1.12.0
- **ML-Agents**: 0.8.1



## 알고리즘

모든 알고리즘은 [파이썬](<https://www.python.org/>)과 [텐서플로](<https://www.tensorflow.org/>)를 통해 작성되었습니다. 알고리즘은 텐서플로 1.5 이상에서 실행이 가능합니다. 제공하는 심층강화학습 알고리즘들은 다음과 같습니다.  

1. **DQN**: 소코반 환경에서 에이전트를 학습하기 위한 Deep Q Network(DQN) 알고리즘입니다 ([Paper](<https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf>)).  
2. **DDPG**: 드론 환경에서 에이전트를 학습하기 위한 Deep Deterministic Policy Gradient (DDPG) 알고리즘입니다 ([Paper](<https://arxiv.org/abs/1509.02971>)). 
3. **DQN_Adversarial**: 두개의 적대적인 에이전트를 학습하기 위한 DQN 알고리즘입니다. . 각 에이전트를 위한 두개의 DQN가 각각 있어서 에이전트는 상대방을 이기는 방향으로 학습을 수행합니다.  이 알고리즘은 퐁 환경에서 에이전트들의 학습을 위해 사용됩니다. 
4. **DDDQN_Curriculum**: Double DQN ([Paper](<https://arxiv.org/abs/1509.06461>)) 과 Dueling DQN ([Paper](<https://arxiv.org/abs/1511.06581>)) 알고리즘을 적용한 DQN 알고리즘입니다. 소코반 커리큘럼 환경에서 에이전트를 학습시키기 위한 알고리즘입니다. 
5. **Behavioral Cloning (BC)**: Behavioral cloning 알고리즘은 닷지 환경에서 에이전트를 학습하기 위한 알고리즘입니다. 모방 학습의 일종이며 사람의 데이터를 기반으로 지도학습을 통해 에이전트의 정책을 학습합니다. 

   

<br>

## 환경 

모든 환경들은 Unity ML-agents version 0.8을 이용하여 제작되었습니다. 다음과 같은 5개의 데모 환경이 제공됩니다.

### 1. 소코반

<img src="./images/Sokoban.gif" alt="Sokoban" style="width: 500px;"/>



### 2. 드론 

드론 환경에서는 [ProfessionalAssets](https://assetstore.unity.com/publishers/31857)에서 제작한 다음의 에셋을 사용하였습니다. 

- [Free Drone Pack](https://assetstore.unity.com/packages/tools/physics/free-pack-117641)

- [Professional Drone Pack](https://assetstore.unity.com/packages/tools/physics/professional-drone-pack-drone-controller-vr-pc-mobile-gamepad-100970): 위 드론 에셋의 유료 버전입니다. 더 많은 환경과 드론들을 제공합니다.

<img src="./images/Drone.gif" alt="Drone" style="width: 500px;"/>

### 3. 퐁

<img src="./images/Pong.gif" alt="Pong" style="width: 500px;"/>



### 4. 소코반 커리큘럼

<img src="./images/Sokoban_Curriculum.gif" alt="Sokoban_Curriculum" style="width: 500px;"/>



### 5. 닷지

<img src="./images/Dodge.gif" alt="Dodge" style="width: 500px;"/>









