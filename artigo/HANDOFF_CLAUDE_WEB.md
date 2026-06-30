# Pacote de handoff para o Claude Web (Overleaf, template SBC)

Este documento é o material a ser enviado ao **Claude Web**, que vai navegar no
navegador (Brave) até o **Overleaf** (projeto já criado com o **template SBC**) e montar o
artigo final. O conteúdo do artigo já está pronto e revisado; a tarefa do Claude Web é
transferi-lo para o Overleaf e compilar.

---

## Objetivo

Montar, no projeto Overleaf existente (template SBC), o artigo científico:

> **Identificação de Fatores Relevantes para Melhoria de Ranqueamento em SERP a partir de Métricas de SEO**
> Autor: Felipe Augusto Medici de Oliveira (RA 2458772), UTFPR Campus Campo Mourão.

O artigo está em **português**, com escrita científica, sem travessões de IA, e contém todas
as seções exigidas. Ele já cumpre os requisitos: resumo com 188 palavras e abstract com 172
palavras (limite de 250), 18 referências (mínimo de 15), 8 seções obrigatórias, tabelas e
figuras com formatação para publicação.

## Arquivos do pacote (na pasta `artigo/`)

1. `artigo_sbc.tex` — corpo completo do artigo em LaTeX, no formato SBC.
2. `referencias.bib` — 18 referências em BibTeX.
3. `figuras/` — 7 figuras usadas no texto, todas em PNG:
   - `class_distribution.png`
   - `target_correlation.png`
   - `model_comparison_prauc.png`
   - `critical_difference.png`
   - `confusion_matrix.png`
   - `roc_pr_curves.png`
   - `group_importance.png`

## Passo a passo no Overleaf

1. Abrir o projeto Overleaf que já contém o **template SBC** (com os arquivos
   `sbc-template.sty` e `sbc.bst` presentes).
2. Substituir o conteúdo do arquivo principal `.tex` do projeto pelo conteúdo de
   `artigo_sbc.tex`. Manter no projeto os arquivos do template SBC (`sbc-template.sty` e
   `sbc.bst`); não removê-los.
3. Enviar `referencias.bib` para a raiz do projeto (o `.tex` usa
   `\bibliography{referencias}`).
4. Criar a pasta `figuras/` no projeto e enviar as 7 imagens listadas acima (o `.tex` usa
   `\graphicspath{{figuras/}}`).
5. Definir o compilador como **pdfLaTeX** e a sequência de compilação como
   **LaTeX → BibTeX → LaTeX → LaTeX** (no Overleaf, basta recompilar; o menu de bibliografia
   resolve as citações).
6. Substituir, em dois pontos do texto, a URL provisória
   `https://github.com/usuario/seo-serp-ranking` pelo endereço definitivo do repositório
   GitHub (uma na introdução, em nota de rodapé; outra na seção de disponibilidade).
7. Compilar e revisar o PDF.

## Lista de verificação final

- [ ] O PDF compila sem erros e as citações aparecem resolvidas (sem `[?]`).
- [ ] As 7 figuras aparecem corretamente posicionadas.
- [ ] O artigo tem entre **10 e 12 páginas**, incluindo referências. Caso ultrapasse 12 ou
      fique abaixo de 10, ajustar o tamanho das figuras (parâmetro `width`) e a quebra de
      parágrafos, sem remover seções nem conteúdo.
- [ ] As 8 seções estão presentes: Título/Autor/Resumo/Abstract, Introdução, Trabalhos
      Relacionados, Conjunto de Dados, Materiais e Métodos, Resultados e Discussão, Conclusão,
      Referências.
- [ ] A URL do repositório foi substituída pelo endereço real.
- [ ] O texto permanece em português, sem travessões de IA.

## Observações

- O artigo já inclui a declaração de uso de IA generativa (Spec Driven Development) na seção
  de disponibilidade, conforme exigido pelo evento.
- Não alterar números, tabelas ou resultados: eles foram gerados pelo código em
  `src/experiment.py` e refletem os experimentos executados.
