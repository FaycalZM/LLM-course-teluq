---
title: "Mission 1. La Forge du Savoir"
weight: 1
---

# Évaluation 2 : L'Architecte du Savoir
## Mission 1 : La Forge du Savoir – Ingestion et Chunking Stratégique

Bonjour à toutes et à tous ! Nous entamons aujourd'hui une épreuve de vérité. Vous avez appris à transformer du texte en vecteurs, mais je vais être très directe avec vous : si vous découpez mal votre savoir, votre IA sera comme un érudit qui a lu des pages déchirées au hasard. 

> [!IMPORTANT]
**Je dois insister :** le "Chunking" n'est pas une étape de pré-traitement mineure, c'est l'acte fondateur de votre moteur de recherche. 

Dans cette première mission, nous allons apprendre à ne pas briser le sens au milieu d'une phrase et à marquer chaque morceau d'une empreinte indélébile : les métadonnées. Préparez votre environnement, l'architecture commence ici !

> [!NOTE]
**Note** : La solution complète avec le code et l'analyse approfondie est incluse dans le notebook Colab qui sera partagé avec le prof.
---

## Objectif de la Mission
L'étudiant doit préparer un corpus technique (documentation de bibliothèques d'IA) en utilisant une stratégie de découpage récursive intelligente. Contrairement aux laboratoires précédents, l'overlap (chevauchement) ne sera pas fixé au hasard mais devra être justifié pour préserver la "trace contextuelle" des documents.

---

## 1. Configuration de l'environnement (Setup)
*Copiez ce bloc dans votre première cellule Colab. Assurez-vous d'utiliser l'accélérateur matériel **GPU T4**.*

```python
# Installation des bibliothèques nécessaires à l'architecture hybride
# !pip install -q langchain langchain-community langchain-classic sentence-transformers faiss-cpu rank_bm25

import torch
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.docstore.document import Document

# Hardware Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Audit matériel : Inférence sur {device}")
```

---

## 2. Le Corpus à Ingérer
*Voici unn aperçu des données brutes que vous devez transformer en une base de connaissances organisée.*

<a href="URL-ICI(Knowledge-Base-Simulation.ipynb)" target="_blank" rel="noopener" class="colab-link">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab-link">
</a>

```python
raw_technical_data = [
    {
        "title": "Optimisation du KV Cache",
        "content": "Le cache KV est essentiel pour réduire la latence des LLM. Il stocke les matrices Key et Value des tokens passés. Cependant, sur les séquences longues, il peut occuper plusieurs Go de VRAM, nécessitant des techniques comme la Grouped-Query Attention (GQA).",
        "category": "Architecture",
        "year": 2023
    },
    {
        "title": "La méthode LoRA",
        "content": "Low-Rank Adaptation (LoRA) permet de fine-tuner des modèles massifs en n'entraînant qu'une fraction des paramètres. En décomposant la mise à jour des poids en deux matrices de rang faible, on réduit drastiquement le coût computationnel sans perdre en performance sémantique.",
        "category": "Fine-tuning",
        "year": 2021
    },
    {
        "title": "Défis de l'Anisotropie",
        "content": "L'anisotropie, ou effet de cône, est un problème majeur où les embeddings se regroupent dans une direction étroite. Cela rend la similarité cosinus moins discriminante pour les mots rares, car les scores restent élevés même pour des paires sans rapport.",
        "category": "Embeddings",
        "year": 2019
    }
    # Note : En situation réelle, imaginez ce dictionnaire multiplié par 1000.
]
```

---

## 3. Vos Tâches de Mission

### Tâche 1 : Transformation en Objets "Document"
Vous devez transformer la liste `raw_technical_data` en une liste d'objets `Document` (LangChain).

> [!IMPORTANT]
**Je dois insister :** Chaque document doit impérativement contenir, dans son dictionnaire `metadata`, les clés `source`, `category` et `year`. C'est ce qui permettra le filtrage intelligent dans la Mission 2.


### Tâche 2 : Stratégie de Chunking Récursif
Utilisez le `RecursiveCharacterTextSplitter` avec les contraintes suivantes :
1.  **Taille du chunk** : 150 caractères (volontairement petit pour tester la résilience).
2.  **Overlap** : Calculez dynamiquement l'overlap pour qu'il représente **20%** de la taille du chunk.
3.  **Justification** : Rédigez un court commentaire expliquant pourquoi un overlap de 20% est préférable à un overlap de 0% dans le cadre d'un assistant technique.



### Tâche 3 : Audit de l'Empreinte
Affichez le contenu et les métadonnées du 3ème chunk généré. Vérifiez que la catégorie et l'année ont bien survécu au découpage.

---

## Mes avertissements

> [!WARNING]
>*   **Attention :** Ne confondez pas `chunk_size` (nombre de caractères) et `tokens`. Dans cette mission, nous travaillons sur les caractères.
>*   **Erreur fréquente :** Oublier de passer le `device` au modèle d'embedding plus tard. Préparez bien vos vecteurs pour qu'ils soient en `float32`.

> [!TIP]
**Le conseil de l'expert :** Regardez comment le splitter récursif essaie d'abord de couper sur les doubles retours à la ligne, puis sur les points. C'est cette hiérarchie qui sauve votre cohérence sémantique !


---

**Une fois cette Mission 1 terminée et vos documents découpés, nous passerons à la Mission 2 : l'Indexation Hybride et la fusion de scores RRF.**
