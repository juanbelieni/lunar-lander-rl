  # Reinforcment Learning - Lunar Lander

Repositório dedicado à avaliação final da disciplina de Aprendizado por Reforço na FGV-EMAp. Neste projeto, exploramos abordagens para resolver o problema do Lunar Lander do OpenAI Gym usando Deep Q-Learning e Deep SARSA para o ambiente padrão, além de Deep SARSA para ambientes vetorizados com a possibilidade de randomização de parâmetros.

## O jogo
Lunar Lander é um jogo onde se manobra um módulo lunar para tentar pousá-lo com cuidado em uma plataforma de pouso, fazendo parte dos ambientes Box2D.
  - [Documentação do ambiente do Lunar Lander no Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Ambiente padrão
Em ambiente padrão, o jogo foi testado em dois modelos: Deep Q-Learning (DQN) e Deep Sarsa. 
O código do DQN foi desenvolvido com base em um [exercício de codificação do curso de Deep Reinforcement Learning da udacity](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html). Posteriormente, realizamos as adaptações necessárias para a implementação do modelo SARSA.

Alguns componentes chaves do modelo:
- **Rede Neural Multicamadas:** Utilizada para estimar os valores de ação para cada estado.
- **Buffer de Repetição:** Responsável por armazenar informações dos episódios, permitindo a amostragem e treinamento eficientes da rede.
- **Estratégia Epsilon-Greedy:** Adotada para a exploração, alternando entre a escolha da melhor ação conhecida (greedy) e escolhas aleatórias (exploração).

### Resultados Deep Q-Learning

![Rewards DQN](Ambiente padrão/results/output_dqn.png)

![Vídeo DQN](Ambiente padrão/results/dqn.mp4)

### Resultados Deep Sarsa

![Rewards Sarsa](Ambiente padrão/results/output_sarsa.png)

![Vídeo Sarsa](Ambiente padrão/results/sarsa.mp4)

### Ambiente vetorizado




