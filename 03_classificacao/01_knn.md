# k-Vizinhos mais próximos (k-NN)

O k-NN é o classificador mais intuitivo que existe: para prever a classe de um ponto, olhamos seus $k$ vizinhos mais próximos e votamos. Não há treino propriamente dito — o modelo simplesmente memoriza os dados (aprendizagem preguiçosa).

Sua simplicidade esconde decisões importantes: qual $k$ usar, qual métrica de distância e por que a padronização das variáveis é obrigatória. Também é a porta de entrada para entender a maldição da dimensionalidade.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Efeito de $k$ na fronteira | de irregular (k=1) a suave (k grande) |
| Importância da escala | distâncias com e sem padronização |
