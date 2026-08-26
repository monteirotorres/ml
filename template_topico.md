# Título do tópico

Uma frase de abertura que situa o leitor: o que este tópico cobre e por que importa.

## Motivação

Motivação concreta antes da teoria. Use um exemplo da vida real ou do domínio biológico/médico.

> Uma citação ou insight central em blockquote, se couber.

## Conceito principal

Explique o conceito com palavras antes de qualquer fórmula. Mostre a intuição.

$$
\text{Fórmula central aqui, se houver}
$$

Onde:
- $x$ é a variável de entrada
- $\hat{y}$ é a predição do modelo

## Sub-seção

### Sub-sub-seção (use com moderação)

Tabelas são bem suportadas:

| Coluna A | Coluna B | Coluna C |
| --- | --- | --- |
| Linha 1 | valor | descrição |
| Linha 2 | valor | descrição |

## Exemplo numérico (ou Python snippet)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([2.1, 4.0, 6.2, 7.9])

modelo = LinearRegression().fit(X, y)
print(f"θ₁ = {modelo.coef_[0]:.3f}, θ₀ = {modelo.intercept_:.3f}")
```

## Cuidados e limitações

- Limitação 1
- Limitação 2

## No notebook

O notebook `XX_topico.ipynb` demonstra [o que o notebook cobre]:
- Ponto 1
- Ponto 2

Abra-o no Colab para experimentar os parâmetros e visualizar os resultados.

## Referências

- Referência 1 (autor, ano, título)
- Referência 2
