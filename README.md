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

Como um episódio é considerado solução se obtiver pelo menos 200 pontos, vemos que ambos os modelos chegaram a solução, onde o Sarsa parece ter menos variância, o que pode ser influenciado por sua aprendizagem on-policy, enquanto o DQN é off-policy.

### Resultados Deep Q-Learning


<img src="/Ambiente padrão/results/output_dqn.png">

<img src="/Ambiente padrão/results/dqn.gif">



### Resultados Deep Sarsa

<img src="/Ambiente padrão/results/output_sarsa.png">

<img src="/Ambiente padrão/results/dqn.gif">



## Ambiente vetorizado

Para avançar um pouco além do problema padrão, implementamos uma versão vetorizada do ambiente, onde é possível randomizar alguns parâmetros do ambiente, como o vento, a força do vento e a turbulência.

Neste caso, o modelo utilizado foi o Deep Sarsa, que foi adaptado para receber um batch de estados e retornar um batch de ações. Os ambientes aleatórios são iniciados antes do treinamento e o modelo é treinado para perfomar em todos eles, mas ao longo do treinamento não são gerados novos ambientes aleatórios. 

Também notamos que era melhor o modelo ser treinado em cada epoch verificando menos steps de mais ambientes, do que mais steps de menos ambientes. Dessa forma, precisamos iniciar um número maior de ambientes aleatórios para obter melhores resultados, o que impactou significativamente o tempo de treinamento pelo custo computacional. Por esse motivo, também não implementamos um buffer da forma tradicional, pois seria ainda mais custoso computacionalmente salvar todos os estados de todos os ambientes.

### Resultados Deep Sarsa
<img src="./video/LunarLander-sarsa-random-envs.mp4">

## Apresentação

Slides:

* [PDF](./slides/lunar_lander_presentation.pdf)
* [PPTX](./slides/lunar_lander_presentation.pptx)



