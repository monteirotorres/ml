# O que é aprendizagem de máquina?

Aprendizagem de máquina é a arte de escrever programas que **melhoram com a
experiência** em vez de seguir regras que alguém escreveu à mão. Neste primeiro
tópico definimos o que isso significa, por que a abordagem é tão poderosa e onde
ela se encaixa dentro da estatística e da inteligência artificial.

▶ [Slides — Fundamentos](fundamentos_slides.html)

## Por que isso importa

Imagine que você precisa escrever um programa para distinguir uma célula tumoral
de uma célula saudável a partir de uma imagem de microscopia. A abordagem
tradicional seria sentar com um patologista e transcrever regras: "se o núcleo
for maior que tal limiar **e** a razão núcleo/citoplasma passar de tanto,
então...". O problema é que essas regras são frágeis, incompletas e impossíveis
de manter quando o número de características cresce.

A aprendizagem de máquina inverte a lógica. Em vez de programar as regras,
mostramos ao computador **muitos exemplos rotulados** — imagens já classificadas
por especialistas — e deixamos que ele **descubra sozinho** o padrão que separa
as duas classes. O programador fornece os dados e a estrutura do modelo; os
detalhes emergem do ajuste aos dados.

> "A aprendizagem de máquina é o campo de estudo que dá aos computadores a
> capacidade de aprender sem serem explicitamente programados." — atribuída a
> Arthur Samuel, 1959.

## Uma definição operacional

A definição mais citada é a de Tom Mitchell (1997), e é útil porque é concreta:

> Diz-se que um programa **aprende** com a experiência $E$, em relação a uma
> tarefa $T$ e uma medida de desempenho $P$, se seu desempenho em $T$, medido por
> $P$, melhora com a experiência $E$.

No exemplo da patologia:

- **Tarefa $T$**: classificar uma célula como tumoral ou saudável.
- **Experiência $E$**: um conjunto de imagens já rotuladas.
- **Desempenho $P$**: a proporção de células classificadas corretamente.

Se ao ver mais imagens rotuladas o programa acerta mais, então ele está
aprendendo no sentido técnico. Essa tríade $(T, E, P)$ é um bom checklist para
qualquer projeto: se você não consegue nomear as três, ainda não formulou o
problema.

## Aprender = ajustar uma função

Em termos matemáticos, quase todo problema de aprendizagem supervisionada busca
uma função $f$ que mapeia entradas em saídas:

$$
\hat{y} = f(\mathbf{x}; \boldsymbol\theta)
$$

Onde:

- $\mathbf{x}$ é o vetor de **características** (features) de um exemplo — por
  exemplo, medidas extraídas de uma célula;
- $\hat{y}$ é a **predição** do modelo;
- $\boldsymbol\theta$ são os **parâmetros** que o algoritmo ajusta a partir dos
  dados.

"Aprender" é encontrar os valores de $\boldsymbol\theta$ que fazem $\hat{y}$ ficar
o mais próximo possível dos valores verdadeiros $y$ nos exemplos observados —
**e**, crucialmente, em exemplos que o modelo ainda não viu. Essa segunda parte,
a **generalização**, é o que separa a aprendizagem de máquina de uma mera tabela
de consulta, e é o tema central do curso.

## Onde a ML se encaixa

Vale situar os termos que costumam ser confundidos:

| Termo | O que é |
| --- | --- |
| Inteligência Artificial (IA) | Campo amplo de fazer máquinas resolverem tarefas "inteligentes". |
| Aprendizagem de Máquina (ML) | Subárea da IA que aprende padrões a partir de dados. |
| Deep Learning | Subárea da ML baseada em redes neurais profundas. |
| Estatística | Base teórica de boa parte da ML; foca em inferência e incerteza. |

A ML e a estatística clássica compartilham quase toda a matemática. A diferença
é de ênfase: a estatística tradicional prioriza **explicar** e testar hipóteses
sobre um mecanismo; a ML prioriza **prever** bem em dados novos, mesmo que o
modelo seja difícil de interpretar. Nenhuma das duas é superior — elas respondem
a perguntas diferentes.

## Quando (não) usar ML

Aprendizagem de máquina não é a resposta para tudo. Ela brilha quando:

- existe um padrão a ser aprendido;
- esse padrão é difícil de expressar em regras explícitas;
- há **dados** suficientes e representativos.

Se o problema pode ser resolvido com uma regra simples e exata (converter
Celsius em Fahrenheit, por exemplo), um modelo de ML é só complicação
desnecessária. E se não há dados de qualidade, nenhum algoritmo faz milagre — o
lema **"garbage in, garbage out"** é impiedoso.

## No notebook

O notebook `01_introducao_ml.ipynb` demonstra na prática o ciclo mínimo de um
projeto de ML:

- carregar o conjunto de dados Iris;
- treinar um classificador simples em poucas linhas de scikit-learn;
- medir o desempenho em dados que o modelo não viu.

Abra-o no Colab para ver o "olá, mundo" da aprendizagem supervisionada.

## Exercícios

**1.** Para o problema "prever se um paciente será readmitido no hospital em 30
dias", identifique a tarefa $T$, a experiência $E$ e a medida de desempenho $P$.

<details><summary>Ver solução</summary>

- **$T$**: classificar cada paciente que recebe alta como "será readmitido em 30
  dias" ou "não será".
- **$E$**: registros históricos de altas hospitalares, cada um com as
  características do paciente e o desfecho real (readmitido ou não).
- **$P$**: uma métrica de classificação adequada ao problema — como a AUC ou o
  recall na classe "readmitido", já que deixar de identificar uma readmissão
  costuma ser mais custoso do que um alarme falso.

</details>

**2.** Explique por que uma "tabela de consulta" que memoriza todos os exemplos
de treino não é considerada aprendizagem de máquina útil.

<details><summary>Ver solução</summary>

Uma tabela de consulta acerta 100% nos exemplos que memorizou, mas não tem
nenhuma capacidade de responder a uma entrada **nova** que não esteja na tabela.
O objetivo da ML é **generalizar** — capturar o padrão subjacente para prever bem
em dados nunca vistos. Um modelo que só memoriza está, na verdade, no extremo do
overfitting (tópico 4).

</details>

## Referências

- Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, cap. 1.
- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 1–2.
