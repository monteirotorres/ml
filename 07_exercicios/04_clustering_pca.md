# Exercícios — Clustering e PCA

Lista de aprendizagem não supervisionada: k-means, clustering hierárquico, escolha
do número de grupos, redução de dimensionalidade com PCA e visualização. As
soluções em `04_clustering_pca.ipynb` enfatizam a **interpretação** — clusters não
vêm com rótulos, e cabe ao analista dar-lhes sentido.

## Exercício 1 — Cotovelo e silhueta

No `wine` (padronizado), trace a curva do cotovelo e a silhueta média para $k$ de 2
a 8. Os dois métodos concordam? Qual $k$ você escolheria?

<details><summary>Ver resposta</summary>

A inércia cai sempre; procura-se o "joelho". A silhueta tem um pico. Eles nem sempre
concordam exatamente — quando divergem, vale inspecionar os grupos e usar
conhecimento do domínio. No `wine`, que tem 3 cultivares, um $k$ próximo de 3 costuma
ser defensável.

</details>

## Exercício 2 — Padronizar antes de agrupar

Rode o k-means no `wine` **sem** padronizar e depois **com** padronização. Compare
os grupos com os rótulos reais (índice de Rand ajustado). Por que a diferença é
grande?

<details><summary>Ver resposta</summary>

Sem padronizar, variáveis de escala grande (como `proline`, na casa dos milhares)
dominam a distância euclidiana, e os grupos se formam quase só por elas — ARI baixo.
Padronizando, todas as variáveis pesam de forma comparável e os grupos passam a
refletir a estrutura real — ARI bem maior. Em métodos de distância, padronizar não é
opcional.

</details>

## Exercício 3 — Quantas componentes reter?

Aplique PCA ao `breast cancer` (padronizado) e descubra quantas componentes são
necessárias para reter 95% da variância. Compare com as 30 variáveis originais.

<details><summary>Ver resposta</summary>

Tipicamente cerca de **10 componentes** retêm 95% da variância das 30 originais — uma
compressão de 3× perdendo só 5% da variação. Mostra o quanto há de redundância
(variáveis correlacionadas) nos dados, e por que a PCA é útil antes de métodos
sensíveis à dimensão.

</details>

## Exercício 4 — PCA antes do t-SNE

No `digits`, compare visualizar com t-SNE aplicado (a) direto nas 64 dimensões e
(b) após reduzir para ~20 componentes com PCA. O que muda? E por que **não** se deve
medir distâncias entre grupos no mapa t-SNE?

<details><summary>Ver resposta</summary>

Aplicar PCA antes acelera o t-SNE e costuma remover ruído, com um mapa de qualidade
parecida ou melhor. Quanto às distâncias: o t-SNE preserva **vizinhança local**, não
distâncias globais — grupos afastados no mapa não são necessariamente mais
diferentes, e o tamanho dos grupos é artefato. O mapa serve para levantar hipóteses,
não para quantificar.

</details>
