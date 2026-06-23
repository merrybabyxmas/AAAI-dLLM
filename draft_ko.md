# 한글 초안: Abstract, Introduction, Related Work, Method

## Abstract

최근 discrete diffusion language model(dLLM)은 autoregressive language model(AR LLM)에 필적하는 성능을 보이며, 새로운 언어 생성 패러다임으로 주목받고 있다. 이에 따라 AR LLM에서 성공적으로 사용된 reinforcement learning(RL) 기법을 dLLM에 적용하려는 시도가 증가하고 있다. 그러나 기존 접근은 대체로 생성된 sequence 전체에 대한 scalar reward를 모든 token 또는 denoising decision에 균등하게 부여하는 방식에 의존한다. 본 논문은 이러한 균등 credit assignment가 dLLM의 생성 구조와 근본적으로 맞지 않는다고 주장한다. AR LLM에서는 token이 고정된 left-to-right 순서로 생성되지만, dLLM에서는 동일한 최종 sequence라도 token들이 어떤 순서와 어떤 상태에서 reveal되었는지에 따라 전혀 다른 denoising trajectory가 형성된다. 따라서 최종 결과가 좋거나 나쁘다는 사실만으로 모든 reveal decision에 동일한 책임을 부여하는 것은 부정확하다. 특히 어떤 block은 이후 trajectory의 방향을 크게 바꾸는 결정적 state transition을 만들 수 있는 반면, 다른 block은 이미 형성된 context를 단순히 완성하는 역할에 그칠 수 있다. 우리는 이 문제를 해결하기 위해 block-level value credit assignment를 제안한다. 제안 방법은 block이 완성되는 시점의 denoising state를 value function으로 평가하고, 인접 block state 간 value difference, 즉 delta-value를 해당 block의 credit으로 사용한다. 이를 통해 최종 reward를 전체 trajectory에 균등하게 분산하는 대신, 실제로 state의 expected outcome을 개선한 block에 더 큰 advantage를 부여한다. LLaDA 기반 dLLM의 block-wise generation 구조를 활용하여 value head를 backbone과 분리해 학습하고, delta-value를 policy update의 advantage로 사용하는 two-stage RL pipeline을 구성한다. 본 연구는 dLLM에서 RL의 핵심이 단순한 sequence-level reward optimization이 아니라, denoising trajectory의 state transition을 평가하고 그 책임을 분해하는 문제임을 보인다.

## Introduction

최근 diffusion language model(dLLM)은 autoregressive language model(AR LLM)의 대안으로 빠르게 부상하고 있다. 기존 AR LLM은 왼쪽에서 오른쪽으로 token을 순차적으로 생성하는 반면, dLLM은 masked sequence를 반복적으로 denoise하면서 여러 위치의 token을 비순차적으로 reveal한다. 이러한 생성 방식은 병렬적이고 유연한 infilling을 가능하게 하며, 최근 LLaDA와 Dream 계열 모델은 instruction following, mathematical reasoning, code generation 등 다양한 benchmark에서 경쟁력 있는 성능을 보이고 있다. 자연스럽게 다음 질문은 dLLM 역시 AR LLM처럼 reinforcement learning(RL)을 통해 더 강화할 수 있는가이다. 실제로 최근 연구들은 GRPO 또는 DDPO 계열의 학습 방식을 dLLM에 적용하여, 최종 generation의 reward를 기반으로 policy를 업데이트하려는 시도를 보이고 있다. 그러나 우리는 dLLM에 AR식 RL을 그대로 이식하는 것은 credit assignment 측면에서 근본적인 한계를 가진다고 본다.

AR LLM에서 trajectory는 비교적 명확하다. 모델은 token을 left-to-right 순서로 생성하고, 각 action은 이전 prefix에 조건부로 다음 token을 선택하는 행위로 해석된다. 따라서 최종 reward를 sequence-level 또는 group-level advantage로 변환해 token-level log-probability에 적용하는 방식은 적어도 생성 순서의 관점에서는 자연스럽다. 반면 dLLM에서는 동일한 최종 output이라도 수많은 서로 다른 denoising trajectory를 통해 생성될 수 있다. 어떤 token이 먼저 reveal되었는지, 그 token이 reveal될 당시 주변 context가 얼마나 masked되어 있었는지, 해당 decision이 이후 reveal 과정에 어떤 제약을 만들었는지에 따라 각 decision의 책임은 크게 달라진다. 그럼에도 기존 RL 적용 방식은 최종 sequence가 좋으면 모든 token decision을 좋은 행동으로, 최종 sequence가 나쁘면 모든 token decision을 나쁜 행동으로 간주하는 경향이 있다. 이는 dLLM의 비순차적 생성 구조를 고려하지 못한 과도하게 단순한 reward broadcasting이다.

본 논문은 다음 질문에서 출발한다. 좋은 결과는 모두 좋은 행동들로 이루어져 있는가? 나쁜 결과는 모두 나쁜 행동들로 이루어져 있는가? 우리는 dLLM에서는 그렇지 않다고 주장한다. dLLM의 denoising trajectory를 보면, 각 reveal decision의 책임은 그것이 발생한 state의 자유도와 이후 trajectory에 미친 영향에 따라 달라진다. 초·중반 denoising에서는 많은 position이 아직 masked 상태이며, 모델은 불완전한 context 속에서 어떤 token 또는 block을 확정해야 한다. 이때의 decision은 이후 context를 형성하고, 뒤따르는 token들의 가능 공간을 크게 제한할 수 있다. 반면 후반 denoising에서는 이미 많은 token이 reveal되어 있으며, 남은 masked position은 앞서 형성된 context에 의해 강하게 제약되는 경우가 많다. 물론 모든 task에서 후반 token의 책임이 항상 작다고 가정할 수는 없다. 중요한 것은 특정 decision이 초반인지 후반인지가 아니라, 그 decision이 부분적으로 reveal된 state의 가치를 얼마나 변화시켰는가이다. 따라서 dLLM의 credit assignment는 token 위치나 최종 reward만으로 결정되어서는 안 되며, denoising state transition의 가치 변화를 기반으로 정의되어야 한다.

이러한 관찰을 바탕으로 우리는 block-level value credit assignment를 제안한다. 제안 방법은 개별 token에 동일한 reward를 부여하는 대신, dLLM의 block-wise generation 구조에 맞춰 denoising state의 가치를 추정한다. 구체적으로 LLaDA와 같이 generation을 block 단위로 진행하는 dLLM에서, block b가 완성된 시점의 state s_b를 해당 prefix 전체의 hidden representation으로 정의한다. 이후 backbone의 마지막 hidden state를 mean pooling하여 소형 value head에 입력하고, scalar value V(s_b)를 예측한다. 각 block의 credit은 인접 state 간 value 차이로 계산된다. 즉 block b가 완료된 후 value가 얼마나 상승했는지를 통해, 해당 block이 trajectory의 expected outcome을 얼마나 개선했는지 측정한다. 마지막 block에서는 최종 reward와 직전 state value 사이의 temporal-difference error를 사용한다. 이렇게 계산된 delta-value는 해당 block에 포함된 token들의 advantage로 사용된다. 이 방식은 최종 reward를 전체 token에 균등하게 분산하는 대신, 실제로 trajectory의 품질을 개선한 state transition에 더 큰 credit을 부여한다.

우리의 접근은 dLLM에서 RL의 핵심이 단순한 reward maximization이 아니라, denoising trajectory에 대한 올바른 책임 분해임을 강조한다. value function은 어떤 block이 trajectory를 실질적으로 개선했는지, 어떤 block이 이미 결정된 결과를 단순히 완성했는지를 데이터로부터 학습한다. 이를 위해 우리는 value head와 policy update를 분리한 two-stage training pipeline을 구성한다. 먼저 frozen backbone 위에서 value head를 Huber loss로 학습하여 각 block state의 expected reward를 예측하게 한다. 이후 detach된 delta-value를 advantage로 사용하여 PPO-style clipped objective와 KL penalty로 policy를 업데이트한다. 이 설계는 value learning이 backbone gradient를 직접 오염시키지 않도록 하며, dLLM의 block-wise denoising 구조에 맞는 안정적인 RL 학습을 가능하게 한다. 본 논문의 기여는 세 가지이다. 첫째, dLLM RL에서 sequence-level reward broadcasting이 부적절한 이유를 denoising trajectory와 action responsibility 관점에서 분석한다. 둘째, block state value의 temporal difference를 이용한 block-level credit assignment 방법을 제안한다. 셋째, value head update와 policy update를 분리한 실용적인 dLLM RL pipeline을 제시한다.

## Related Work

Diffusion language model은 discrete token sequence를 autoregressive하게 생성하는 대신, masked 또는 noised sequence를 반복적으로 denoise하는 방식으로 언어 생성을 수행한다. 초기 연구인 D3PM은 discrete state-space에서의 denoising diffusion framework를 제시했으며, Diffusion-LM과 DiffuSeq는 continuous 또는 sequence-to-sequence setting에서 diffusion 기반 text generation의 가능성을 보였다. 이후 SSD-LM은 simplex representation과 semi-autoregressive generation을 결합했고, SEDD와 MDLM은 discrete diffusion 또는 masked diffusion objective를 정교화하여 language modeling 성능을 개선했다. 최근에는 LLaDA와 Dream과 같은 large-scale diffusion language model이 등장하면서 dLLM이 단순한 text infilling 모델을 넘어 instruction following, reasoning, coding 등 일반 LLM benchmark에서도 경쟁력 있는 성능을 보일 수 있음이 보고되었다. 또한 Block Diffusion은 AR과 diffusion 사이의 중간적 생성 구조를 제안하여, 순차성과 병렬성 사이의 trade-off를 다루었다. 한편 DAWN과 같은 최근 inference 연구는 dLLM에서 어떤 위치를 언제 reveal할 것인지가 generation quality와 efficiency에 중요한 영향을 미친다는 점을 보여준다. 이러한 연구들은 dLLM의 학습 objective, scaling, sampling efficiency, inference scheduling, generation quality를 주로 다루었다. 반면 본 연구는 dLLM의 RL fine-tuning 과정에서 발생하는 credit assignment 문제에 초점을 맞춘다. 특히 우리는 최종 output의 품질뿐 아니라, 그 output이 어떤 denoising state transition을 통해 생성되었는지가 policy learning에서 핵심적인 정보라고 본다. 따라서 본 연구는 새로운 dLLM architecture나 diffusion objective를 제안하는 것이 아니라, dLLM의 비순차적 generation trajectory에 맞는 reward-to-action credit decomposition을 제안한다는 점에서 기존 dLLM 연구와 구분된다.

대규모 언어 모델의 RL 학습에서는 sequence-level reward를 token-level policy gradient에 적용하는 방식이 널리 사용되어 왔다. RLHF, PPO, DPO, GRPO 계열 방법은 주로 autoregressive generation을 전제로 하며, 생성된 response의 전체 reward 또는 group-relative advantage를 각 token log-probability에 적용한다. 최근에는 이러한 방법을 diffusion model 또는 dLLM에 적용하려는 시도도 등장하고 있다. DDPO 계열 연구는 diffusion model의 denoising process를 multi-step decision-making 문제로 보고 reward-weighted policy optimization을 수행하며, d1과 같은 dLLM RL 접근은 AR LLM에서 성공한 GRPO-style sequence-level reward를 masked diffusion generation에 적용하려 한다. 그러나 이러한 접근은 denoising step 또는 token reveal decision 간의 책임 차이를 충분히 반영하지 못한다. diffusion model에서는 각 denoising step이 최종 sample의 품질에 미치는 영향이 다르며, dLLM에서는 특히 reveal 시점의 context completeness와 state uncertainty가 action responsibility를 크게 바꾼다. 본 연구는 이 점에서 기존 sequence-level reward broadcasting 방식과 다르다. 우리는 reward를 전체 token에 균등하게 분배하지 않고, block boundary에서 추정된 state value의 변화량을 이용해 block별 credit을 계산한다. 이는 전통적인 temporal-difference learning의 관점을 dLLM denoising trajectory에 맞게 재해석한 것이다. value function과 temporal-difference learning은 RL에서 state-dependent return을 추정하기 위한 고전적인 도구이지만, 기존 dLLM RL 접근은 주로 최종 reward 또는 group-level reward를 그대로 token-level objective에 전달해 왔다. 우리는 value function을 backbone policy와 분리된 value head로 학습하고, delta-value를 policy update의 고정 advantage로 사용함으로써 value estimation과 policy optimization 간의 gradient interference를 줄인다. 결과적으로 본 연구는 dLLM에서 RL을 적용할 때 필요한 핵심 구성 요소가 단순한 reward model이나 PPO objective가 아니라, denoising trajectory의 state transition을 평가하는 value-based credit assignment임을 제안한다.

## Method

### Problem Formulation

Prompt x_p가 주어졌을 때, dLLM은 diffusion sampling을 통해 G개의 completion {x_c^g}_{g=1}^G를 생성한다. 각 completion x_c^g는 reward function에 의해 scalar reward r^g를 받으며, RL fine-tuning의 목표는 sampled completion의 expected reward를 증가시키도록 policy를 업데이트하는 것이다. GRPO-style 학습에서는 같은 prompt에서 생성된 completion group의 reward를 정규화하여 group-normalized return target \bar{A}^g를 계산한다. 기존 dLLM RL 접근은 이 completion-level quantity를 해당 completion의 모든 token 또는 reveal decision에 동일하게 부여한다. 즉 좋은 completion의 모든 token은 동일하게 좋은 행동으로, 나쁜 completion의 모든 token은 동일하게 나쁜 행동으로 취급된다.

우리는 이러한 credit assignment가 dLLM의 denoising 구조에 비해 지나치게 거칠다고 본다. dLLM의 completion은 고정된 left-to-right action sequence로 생성되지 않고, 각 block 또는 token이 서로 다른 masked context에서 reveal되는 denoising trajectory를 통해 생성된다. 따라서 동일한 최종 reward라도 block마다 책임이 다를 수 있다. 본 논문에서는 completion g의 scalar training target을 R^g로 표기하며, 구현에서는 R^g = \bar{A}^g를 사용한다. 우리의 목표는 각 block transition이 부분적으로 reveal된 state의 가치를 얼마나 변화시켰는지를 추정하고, 그 value difference를 해당 block의 credit으로 사용하는 것이다.

### Block-Level State Representation

우리는 LLaDA-style dLLM의 block-wise generation 구조를 사용한다. Block 크기를 L_B, completion 길이를 L_c라고 할 때, 전체 block 수는 B = ceil(L_c / L_B)로 정의한다. Block b가 완성된 시점의 denoising state s_b^g는 prompt와 b+1개 block까지 reveal된 completion prefix를 backbone에 입력했을 때의 마지막 layer hidden states를 mean pooling하여 얻는다.

s_b^g = MeanPool(h_{-1}(x_p \oplus x_c^{g,< (b+1)L_B})) \in R^H.

여기서 h_{-1}은 backbone의 마지막 layer hidden states이고, mean pooling은 sequence dimension에 대해 수행된다. 이 state는 prompt와 현재까지 완성된 completion block들이 주어졌을 때의 model representation이다. 마지막 block이 L_B보다 짧은 경우에는 실제 completion prefix 길이에 맞춰 state를 계산한다.

### Value Head

Block-level state의 expected return을 추정하기 위해 별도의 value head V_phi를 도입한다. Value head는 두 층짜리 MLP로 구성된다.

V_phi(s_b) = W_2 GELU(W_1 s_b + a_1) + a_2.

LLaDA-style backbone을 사용할 경우 hidden size는 4096이고, value head는 4096 -> 256 -> 1 구조를 가진다. Value head는 fp32로 유지하며, backbone policy와 분리된 AdamW optimizer를 사용하여 critic learning rate 5e-6으로 학습한다.

Value head는 backbone과 분리되어 학습된다. Value 계산 시 backbone은 no_grad 상태로 실행되고, hidden state는 value head에 입력되기 전에 detach된다. 따라서 value head update와 policy update는 computation graph를 공유하지 않는다. 이 설계는 policy gradient가 value head를 거쳐 backbone으로 흘러들어가는 것을 방지하고, value estimation과 policy optimization을 분리한다. Value head의 학습 target은 completion-level target R^g이며, sampled block-boundary states에 대해 Huber loss를 사용한다.

L_V = (1 / (G |B_sample|)) sum_g sum_{b in B_sample} Huber(V_phi(s_b^g), R^g).

### Delta-V Credit Assignment

모든 block boundary에서 V_phi(s_b)를 계산하면 block 수만큼 추가 forward pass가 필요하므로 메모리와 계산 비용이 커진다. 이를 줄이기 위해 최대 K개의 block boundary만 균등 간격으로 subsample한다.

B_sample = {b_0, b_1, ..., b_{K-1}} subset {0, 1, ..., B-1}.

여기서 b_0 = 0과 b_{K-1} = B-1은 항상 포함한다. 각 value forward 결과는 계산 직후 CPU memory로 이동시켜 GPU peak memory를 줄인다.

Non-terminal segment의 block credit은 인접 sampled boundary 사이의 value difference로 정의한다. Terminal block의 경우 completion-level target과 마지막 state value 사이의 TD(0) error를 사용한다.

만약 block b가 [b_i, b_{i+1}) segment에 속하면 delta_b^g = V_phi(s_{b_{i+1}}^g) - V_phi(s_{b_i}^g)로 계산한다. 마지막 block b = B-1의 경우 delta_b^g = R^g - V_phi(s_{B-1}^g)를 사용한다. 같은 sampled segment에 속한 block들은 동일한 delta-value를 공유한다. 이 approximation은 모든 boundary를 계산하지 않고도 trajectory의 구간별 value improvement를 추정하게 해준다.

### Per-Token Advantage

Block-level credit을 token-level policy loss에 사용하기 위해, 각 token은 자신이 속한 block의 delta-value를 기본 advantage로 받는다. 여기에 reveal 순간의 model confidence를 사용한 responsibility weight를 곱한다. Token i가 reveal될 때 모델이 해당 token에 부여한 softmax probability를 c_i^g라고 하자. Confidence-based responsibility weight는 다음과 같이 정의한다.

rho_i^g = (c_i^g + epsilon)^(-alpha).

이 식은 낮은 confidence에서 reveal된 token에 더 큰 weight를 부여한다. 낮은 confidence decision은 불확실성이 큰 상태에서 이루어진 선택일 가능성이 높고, 따라서 더 큰 책임을 가질 수 있기 때문이다. 이후 weight를 completion mask M^g 내부 평균으로 normalize한다.

rho_tilde_i^g = rho_i^g / mean_{j in M^g}(rho_j^g).

최종 per-token advantage는 다음과 같다.

A_i^g = delta_{b(i)}^g * rho_tilde_i^g.

여기서 b(i)는 token i가 속한 block index이다. 만약 value head가 아직 충분히 학습되지 않아 delta-value의 표준편차가 delta_gate보다 작다면, delta-value가 유의미한 credit signal을 제공하지 못한다고 판단한다. 이 경우 fallback으로 기존 group-level target R^g에 confidence weight만 적용한다.

A_i^g = R^g * rho_tilde_i^g.

### Policy Update

Per-token advantage A_i^g를 이용해 PPO-style clipped objective를 계산한다. 현재 policy와 old policy가 token i에 부여한 log-probability를 각각 ell_{theta,i}^g와 ell_{old,i}^g라고 하면 likelihood ratio는 다음과 같다.

q_i^g = exp(ell_{theta,i}^g - ell_{old,i}^g).

Completion mask M^g는 EOS 이전의 completion token positions로 정의한다. Policy loss는 completion mask 내부 token에 대해 clipped surrogate objective를 평균한 뒤 KL penalty를 더해 계산한다.

L_pi = - mean_{g, i in M^g} min(q_i^g A_i^g, clip(q_i^g, 1-eps_clip, 1+eps_clip) A_i^g) + beta D_KL.

A_i^g는 policy update 과정에서 상수로 취급하며, value computation이나 confidence weight를 통해 gradient가 흐르지 않도록 detach한다.

### Training Loop Summary

각 training step은 다섯 단계로 구성된다. 첫째, 현재 policy로 G개의 completion을 generate하고 각 token의 reveal-time confidence c_i^g를 기록한다. 둘째, reward function이 각 completion에 scalar reward r^g를 부여하고, 이를 바탕으로 group-normalized target R^g = \bar{A}^g를 계산한다. 셋째, credit assignment module은 최대 K개의 sampled block-boundary state에 대해 V_phi(s_b)를 계산하고, value outputs을 CPU로 옮긴 뒤 delta-value를 per-token advantage로 변환한다. 넷째, backbone을 frozen/no_grad 상태로 두고 hidden state를 detach한 뒤 value head를 Huber loss로 업데이트한다. 다섯째, detach된 per-token advantage를 사용하여 PPO-style clipped policy loss와 KL penalty로 policy를 업데이트한다.

기본 hyperparameter는 block_length L_B = 32, num_blocks B = 8, max_value_states K = 4, value_hidden_size = 256, critic_lr = 5e-6, delta_v_gate = 0.01, credit_alpha = 1.0, beta = 0.04로 설정한다.
