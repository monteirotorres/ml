# Funções de ativação

As funções de ativação introduzem não linearidade — sem elas, uma rede profunda colapsaria em um único modelo linear. A escolha da ativação afeta diretamente a capacidade de aprendizado e a estabilidade do treino.

Compararemos sigmoide, tanh e ReLU (e suas variantes), discutindo o problema do gradiente que desaparece e por que a ReLU se tornou o padrão nas camadas ocultas das redes modernas.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Sigmoide, tanh, ReLU | formas, derivadas e saturação |
| Gradiente que desaparece | por que a ReLU ajuda |
