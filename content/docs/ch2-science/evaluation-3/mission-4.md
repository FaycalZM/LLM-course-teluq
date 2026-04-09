---
title: "Mission 4. The Verdict of Trust"
weight: 
---


# Évaluation 3 : Le Gardien de la Vérité
## Mission 4 : The Verdict of Trust – Quantitative Audit with Ragas

Bonjour à toutes et à tous ! Nous voici arrivés au sommet de notre système RAG. Votre assistant est capable de chercher, de filtrer et de rédiger avec des citations. C’est une prouesse technique, mais en tant qu'ingénieurs, nous ne pouvons pas nous contenter d'un "ça a l'air de marcher". 

> [!IMPORTANT]
**Je dois insister :** l'intuition est le premier pas vers l'erreur. Aujourd'hui, nous allons devenir des auditeurs scientifiques. 

Nous allons utiliser l'IA pour juger l'IA et transformer la notion floue de "qualité" en indicateurs mathématiques précis. Prêt·e·s pour le verdict final ? 

---

## Objectif de la Mission
L'étudiant doit clôturer son projet en mettant en place un pipeline d'évaluation automatisé. L'enjeu est d'utiliser le framework **Ragas** (Retrieval Augmented Generation Assessment) pour mesurer deux indicateurs critiques :
1.  **Faithfulness (Fidélité)** : La réponse est-elle mathématiquement ancrée dans les documents récupérés ?
2.  **Answer Relevancy (Pertinence)** : La réponse répond-elle directement à l'intention de l'utilisateur ?

---

## Les Concepts de l'Audit : La Triade Ragas

Pour cette mission finale, vous allez mobiliser le concept de **LLM-as-a-judge** (L'IA comme juge). Comme nous l'avons vu en Section 9.3, les métriques classiques (comme BLEU ou ROUGE) sont incapables de détecter une hallucination. Ragas, lui, décompose la réponse en affirmations et les compare aux preuves.

### 1. Mesurer la Fidélité (Faithfulness)
Ce score (entre 0 et 1) vérifie si le LLM n'a pas "inventé" des faits en dehors du contexte fourni. 
*   *Exemple* : Si le document dit "le prix est de 5$" et que l'IA répond "le prix est très bas (environ 4$)", le score chutera car l'affirmation "4$" n'est pas supportée par la source.


### 2. Mesurer la Pertinence (Answer Relevancy)
Cette métrique évalue si la réponse est concise et centrée sur la question. Elle pénalise les réponses trop vagues ou celles qui incluent des informations inutiles non demandées.

---

## 1. Configuration de l'Audit (Setup)
*Installez les outils de mesure. Notez que Ragas utilise un modèle "Critique" pour évaluer vos sorties.*

```python
# Installation of the evaluation framework
# !pip install -q ragas datasets langchain-openai openai

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate, RunConfig
from datasets import Dataset
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import torch
from google.colab import userdata
import os

# NOTE: In a real production environment, we use a very powerful model 
# (like GPT-4o or Claude 3.5) as a judge to evaluate a smaller model (like Phi-3).
OPENAI_KEY = userdata.get('YOUR_KEY_NAME')
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ── 1. OPENAI GPT JUDGE & EMBEDDINGS ────────────────────
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",   # fast + cheap; swap for "gpt-4o" if needed
    temperature=0,
)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

ragas_llm        = LangchainLLMWrapper(langchain_llm=openai_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=openai_embeddings)


# ── 2. ASSIGN TO METRICS ──────────────────────────────────────────────────────
# Ensure metrics are explicitly assigned to our judge
faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
answer_relevancy.embeddings = ragas_embeddings

print("Juge configuré : GPT-4o est prêt à auditer le pipeline.")

```

---

## 2. Vos Tâches de Mission

### Tâche 1 : Préparation du "Dataset d'Audit"
Pour exécuter une évaluation Ragas, vous devez construire un dictionnaire Python nommé `data_samples` contenant exactement quatre colonnes :
*   `question` : La liste des questions posées.
*   `answer` : La liste des réponses générées par votre fonction `generate_grounded_answer` (Mission 3).
*   `contexts` : La liste des listes de textes récupérés par votre `Retriever` (Mission 2).
*   `ground_truth` : La réponse idéale attendue (la "vérité terrain").

### Tâche 2 : Exécution de la Mesure
1.  Transformez votre dictionnaire en un objet `Dataset` Hugging Face.
2.  Lancez la fonction `evaluate()` en utilisant uniquement les métriques `faithfulness` et `answer_relevancy`.

> [!WARNING]
>3.  **Attention technique** : L'évaluation peut être gourmande en appels API. Pour ce laboratoire, limitez-vous à auditer les 2 cas de test vus en Mission 3.

### Tâche 3 : Analyse Critique et Loi de Goodhart
Rédigez une conclusion de 200 mots en répondant à cette problématique : 
*   « Si mon score de **Faithfulness** est de 1.0 mais que mon score de **Answer Relevancy** est de 0.2, quel est le problème concret rencontré par l'utilisateur ? »

> [!TIP]
**Mon conseil** : Repensez à la **Loi de Goodhart** : comment un ingénieur pourrait-il "tricher" pour avoir un score de fidélité parfait au détriment de l'utilité ?

---

## Mes avertissements

> [!WARNING]
>*   **Erreur fréquente :** Fournir le mauvais format à la colonne `contexts`. Ragas attend une **liste de listes** de chaînes de caractères (chaque question peut avoir plusieurs documents sources).
>*   **Le biais du juge :** Rappelez-vous qu'un juge automatique reste une IA. Si vos scores vous semblent absurdes, procédez à une vérification humaine sur un échantillon.

> [!TIP]
**Mon message :** Vous ne terminez pas un code, vous validez un contrat de confiance. Un score Ragas est la signature de votre intégrité en tant que Data Scientist.

---

**Une fois cette Mission 4 terminée, votre notebook Évaluation 3 sera complet. Il représentera un pipeline RAG industriel, sécurisé, optimisé et audité. Félicitations !**