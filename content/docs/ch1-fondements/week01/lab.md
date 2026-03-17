---
title: "Laboratoire"
weight: 6
---

Félicitations ! Vous avez traversé la jungle de l'histoire du NLP. Maintenant, il est temps de mettre les mains dans le cambouis (ou plutôt dans les tokens !). Ce premier laboratoire est conçu pour ancrer vos intuitions. Ne cherchez pas la perfection immédiate, cherchez à comprendre le "pourquoi" derrière le code. Prêt·e·s ? C'est parti !

---

## 🔹 EXERCICE 1 : Tokenisation manuelle (Niveau basique - Intuition BPE)

**Objectif** : Comprendre comment un algorithme comme le Byte Pair Encoding (BPE) crée des jetons (tokens) à partir de caractères fréquents.

**Description** : Implémentez une logique simplifiée qui identifie la paire de caractères la plus fréquente pour simuler la première étape d'un tokeniseur moderne.

**Code à compléter (Testé sur Colab T4)** :

```python
from collections import Counter

def simple_bpe_step(text):
    # 1. On sépare le texte en caractères (en ajoutant un symbole de fin de mot)
    words = text.split()
    # Création d'une liste de listes de caractères
    token_list = [list(word) + ["</w>"] for word in words]
    
    # 2. Compter les paires de caractères adjacentes
    pairs = Counter()
    for word_tokens in token_list:
        for i in range(len(word_tokens) - 1):
            pairs[word_tokens[i], word_tokens[i+1]] += 1
            
    # 3. Trouver la paire la plus fréquente
    best_pair = max(pairs, key=pairs.get)
    return best_pair, pairs[best_pair]

text_example = "le chat chasse le chien dans le jardin"
pair, freq = simple_bpe_step(text_example)

print(f"La paire à fusionner est : {pair} avec une fréquence de {freq}")
# ATTENDU : La paire ('l', 'e') car "le" apparaît 3 fois.
```
<!-- TODO: add colab link -->

**Attentes** : Expliquez pourquoi fusionner "l" et "e" en un seul token "le" est plus efficace pour le modèle que de les traiter séparément.

---

## 🔹 EXERCICE 2 : Analyse comparative (Niveau intermédiaire - Le mur de la polysémie)

**Objectif** : Démontrer par l'analyse pourquoi les embeddings contextuels ont remplacé le Bag-of-Words.

**Consigne** :

Considérez les deux phrases suivantes :

1. "Je dépose mon argent à la **banque**."
2. "Le pêcheur s'installe sur la **banque** du fleuve."

**Tâches** :

1. Si vous utilisez un modèle **Bag-of-Words**, quelle sera la différence de représentation du mot "**banque**" entre ces deux phrases ?
2. Si vous utilisez un **Transformer (ex: BERT)**, comment l'attention permet-elle de différencier ces deux occurrences ?

<details>
<summary>Voir la réponse</summary>

1. **BoW** : Aucune différence. Le mot "banque" est lié à un index unique. Pour le modèle, la finance et la pêche sont identiques ici.

2. **Transformer** : La self-attention du mot "banque" dans la phrase 1 va se lier au mot "argent", tandis que dans la phrase 2, elle se liera à "fleuve" et "pêcheur". Le vecteur résultant sera différent (contextualisé).
</details>

---

## 🔹 EXERCICE 3 : Recherche historique (Niveau avancé - Jalons NLP)

**Objectif** : Synthétiser l'évolution technologique rapide entre 2012 et 2023.

**Consigne** : À l'aide de la **Figure 1-1** et de la **Figure 1-27**, identifiez trois modèles majeurs et expliquez leur apport.

<details>
<summary>Voir la réponse</summary>

1. **Word2Vec (2013)** : Passage des comptes de mots aux vecteurs denses (géométrie du langage).

2. **Transformer (2017)** : Abandon de la lecture séquentielle pour l'attention parallèle.

3. **GPT-3 (2020)** : Démonstration que le passage à l'échelle massive (175B paramètres) permet l'apprentissage sans exemples (Zero-shot).
</details>

---

**Mots-clés de la semaine** : NLP, Bag-of-Words, Embeddings, RNN, LSTM, Attention, Self-Attention, Transformer, Pre-training, Fine-tuning, LLM.

**En prévision de la semaine suivante** : La semaine prochaine, nous plongerons dans les tokens et embeddings — les briques fondamentales que chaque LLM manipule en coulisses.
