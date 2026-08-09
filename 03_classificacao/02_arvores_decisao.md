# Árvores de decisão

Uma árvore de decisão classifica fazendo uma sequência de perguntas do tipo "a variável X é maior que t?", particionando o espaço em regiões. O resultado é um modelo altamente interpretável, que se lê como um fluxograma clínico.

Estudaremos os critérios de divisão (impureza de Gini e entropia), como a profundidade controla a complexidade e por que árvores isoladas tendem a overfittar — motivação direta para os ensembles do capítulo seguinte.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Impureza de Gini | como a árvore escolhe cada divisão |
| Profundidade × overfitting | poda e `max_depth` |
