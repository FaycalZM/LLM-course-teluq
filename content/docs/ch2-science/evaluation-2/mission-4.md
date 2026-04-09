---
title: "Mission 4. Le Verdict de la Précision"
weight: 4
---
{{< katex />}}

# Évaluation 2 : L'Architecte du Savoir
## Mission 4 : Le Verdict de la Précision – Métriques MRR, NDCG et Arbitrage Technique

Bonjour à toutes et à tous ! Nous voici au sommet de notre pyramide. Vous avez construit un moteur hybride (Mission 2) et vous l'avez affiné avec un scalpel de précision (Mission 3). Le système "tourne", certes. Mais est-il réellement bon ? 

> [!IMPORTANT]
**Je dois insister :** en ingénierie des LLM, l'intuition est votre pire ennemie. Ce n'est pas parce que le système a bien répondu à *votre* question qu'il fonctionnera pour des milliers d'utilisateurs. 

Aujourd'hui, nous allons devenir des auditeurs. Nous allons transformer la qualité sémantique en chiffres froids et indiscutables : le **MRR** et le **NDCG**. Respirez, nous passons du rôle d'artisan à celui de scientifique de la mesure !

---

## Objectif de la Mission
L'étudiant doit clore son notebook en implémentant un banc d'essai (*Benchmark*) quantitatif. L'enjeu est de comparer rigoureusement deux versions du pipeline : la recherche hybride simple (RRF) et la recherche avec reranking. L'étudiant devra calculer et interpréter les scores de performance pour justifier l'investissement en temps de calcul du Cross-Encoder.

---

## Les Concepts de Mesure : Juger l'Ordre et la Qualité

Pour cette mission finale, vous devez mobiliser les outils d'évaluation du domaine de l' *Information Retrieval* (IR) détaillés dans le chapitre 8.

### 1. La Test Suite
Comme l'illustre la **Figure 8-16 : Composantes de l'évaluation** , vous allez créer un environnement d'audit. 

{{< bookfig src="187.png" week="09" >}}

*   **Les Requêtes de Test** : Un échantillon représentatif des questions des utilisateurs.
*   **La Vérité Terrain (Ground Truth)** : L'identifiant (ID) exact du document qui détient la réponse.

> [!NOTE]
**Note** : Dans cette mission, nous simplifions la réalité en considérant qu'il n'y a qu'une seule "bonne réponse" par question pour faciliter le calcul du MRR.

### 2. Le MRR (Mean Reciprocal Rank)
Le **MRR** est la métrique de l'efficacité immédiate. Sa logique est décomposée comme suit :
*   Si le bon document est en 1ère position, le score est $1/1 = 1$.
*   S'il est en 2ème position, le score chute à $1/2 = 0.5$.
*   S'il est en 10ème position, le score est de $0.1$.


**L'intuition technique** : Le MRR punit sévèrement les systèmes qui obligent l'utilisateur à scroller. Il mesure la probabilité que le premier lien soit le bon.


### 3. Le NDCG (Normalized Discounted Cumulative Gain)
Le **NDCG** est plus subtil que le MRR. 
*   Il ne se contente pas de vérifier si la réponse est là ; il évalue la **qualité globale du classement**. 
*   Il utilise un facteur de "remise" (*discount*) logarithmique : plus vous descendez dans la liste, moins un document pertinent a de valeur pour le score final. 

> [!IMPORTANT]
**Je dois insister :** Le NDCG est la métrique préférée des moteurs de recherche comme Google ou Bing car elle reflète la satisfaction réelle d'un utilisateur face à une liste de résultats.

---

## 1. Snippets de configuration (Données de Benchmark)
*Copiez ces données dans votre notebook. Elles serviront de référence pour votre audit.*

```python
import time
import pandas as pd
import numpy as np

# --- DONNÉES DE BENCHMARK ---
# Notre "Vérité Terrain" : Question -> Index du chunk contenant la réponse
test_suite = [
    {"query": "Comment gérer les séquences longues dans le cache KV ?", "expected_id": 0},
    {"query": "Fine-tuning efficace en paramètres avec LoRA", "expected_id": 2},
    {"query": "Pourquoi la similarité cosinus échoue-t-elle pour les mots rares ?", "expected_id": 4}
]

# Note : 'expected_id' correspond à l'index original dans la liste 'chunks' de la Mission 1.
```

---

## 2. Vos Tâches de Mission

### Tâche 1 : Implémentation du MRR
Écrivez une fonction `calculate_mrr(results_list, expected_id)` qui renvoie l'inverse du rang du document attendu. Si le document n'est pas dans le Top-K, le score est 0.

### Tâche 2 : Audit comparatif (Le Match)
Vous devez faire s'affronter vos deux architectures sur les 3 requêtes de la `test_suite` :
1.  **Système A** : `hybrid_search` (Mission 2).
2.  **Système B** : `advanced_search_with_rerank` (Mission 3).

Générez un tableau récapitulatif montrant pour chaque requête : le Rang trouvé et le score Reciprocal Rank.

### Tâche 3 : Analyse de la Latence (Le coût de la vérité)
Mesurez le temps moyen d'exécution (via `time.time()`) pour les deux systèmes. 
**L'arbitrage de l'ingénieur** : Rédigez une conclusion technique. Le gain de précision apporté par le reranker (Mission 3) justifie-t-il l'augmentation du temps de réponse ?

---

## Mes avertissements

> [!WARNING]
> *   **Erreur fréquente :** Commencer à compter les rangs à 0. Dans la formule du MRR, la première position est **1**. Si vous divisez par 0, votre code plantera !
> *   **Attention :** Un système peut avoir un excellent MRR sur 3 requêtes et s'effondrer sur 1000. Dans votre conclusion, mentionnez l'importance de la **taille de l'échantillon** pour la validité statistique.

> [!TIP]
**Le conseil de l'expert :** « Regardez au-delà du chiffre. Si le reranker fait passer un document de la position 2 à la position 1, votre MRR double (de 0.5 à 1.0). C'est un saut qualitatif immense pour l'utilisateur, même si cela a coûté 200ms de plus sur le GPU.

---

## Conclusion de l'Évaluation 2

Une fois la Mission 4 terminée, votre notebook devra démontrer :
> 1.  Un découpage de texte qui respecte les métadonnées.
> 2.  Un moteur capable de chercher par sens ET par mots-clés.
> 3.  Une phase de re-classement par attention bidirectionnelle.
> 4.  Une preuve chiffrée que ces étapes améliorent la pertinence.


**Mon message :** 
"Vous avez terminé votre deuxième grande œuvre. Vous n'êtes plus des développeurs qui "tentent" des choses, vous êtes des architectes qui "prouvent" leurs solutions. La semaine prochaine, nous utiliserons ce moteur pour construire un **RAG complet**, où l'IA ne se contentera plus de chercher, mais rédigera des réponses en s'appuyant sur votre architecture. Excellent travail !"