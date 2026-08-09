# t-SNE e UMAP

t-SNE e UMAP são técnicas não lineares de redução de dimensionalidade voltadas à **visualização**. Elas preservam a estrutura local — pontos vizinhos permanecem vizinhos — revelando agrupamentos que a PCA linear pode não capturar.

São padrão em single-cell RNA-seq e outros dados biológicos de altíssima dimensão. Veremos os hiperparâmetros críticos (perplexidade, `n_neighbors`), suas armadilhas de interpretação e por que distâncias globais nesses mapas enganam.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Perplexidade / n_neighbors | efeito no mapa resultante |
| Armadilhas de interpretação | o que NÃO ler nos eixos |
