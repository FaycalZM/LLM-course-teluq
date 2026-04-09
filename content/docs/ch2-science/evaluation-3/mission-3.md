---
title: "Mission 3. The Pillar of Evidence"
weight: 3
---

# Évaluation 3 : Le Gardien de la Vérité
## Mission 3 : The Pillar of Evidence – Grounded Generation and Citations

Bonjour à toutes et à tous ! Nous arrivons à la phase la plus gratifiante, mais aussi la plus périlleuse de notre système RAG. Grâce aux Missions 1 et 2, vous avez un Bibliothécaire hors pair qui rapporte les bons documents. Maintenant, il faut que l'Écrivain (le LLM) rédige la réponse. 

> [!IMPORTANT]
**Je dois insister :** une réponse fluide sans preuve est une menace pour l'intégrité de votre entreprise. 

Aujourd'hui, nous allons apprendre à "brider" l'imagination de l'IA. Nous allons lui imposer une règle de fer : si l'information n'est pas écrite dans les documents fournis, l'IA doit se taire. Mieux encore, elle devra nous dire exactement quelle phrase elle a utilisée pour répondre. Bienvenue dans l'ère de la **génération ancrée**.


---

## Objectif de la Mission
L'étudiant doit concevoir et implémenter la brique de génération finale du système RAG. Le défi est de créer un **Prompt Augmenté** sophistiqué qui force le modèle (Phi-3) à :
1.  Synthétiser l'information en restant strictement fidèle au contexte (Groundedness).
2.  Refuser de répondre si l'information est absente (Anti-hallucination).
3.  Inclure des **citations explicites** (ex: `[Source: title]`) pour chaque fait énoncé.

---

## Les Concepts techniques : Grounding & Citation Logic

Pour transformer une réponse statistique en une réponse documentaire, nous utilisons deux leviers :

### 1. L'Ancrage Sémantique
Regardez la **Figure 9-4 : Recherche sémantique pour la génération**.

{{< bookfig src="197.png" week="09" >}}

*   **Le concept** : On ne donne pas seulement la question au LLM. On injecte les documents trouvés en Mission 2 directement dans le "Prompt Système". 
*   **L'effet** : Le LLM traite ces documents comme sa "vérité temporaire". 

> [!NOTE]
**Note** : C'est ce qu'on appelle la **Génération Conditionnée**. Le modèle n'utilise plus ses poids internes pour inventer, mais ses couches d'attention (Section 3.1) pour résumer.

### 2. L'attribution des sources
La **Figure 9-5 : Réponse RAG avec citations** montre l'étalon-or de la production.

{{< bookfig src="174.png" week="09" >}}

*   **Le mécanisme** : On demande au modèle, via le prompt, d'ajouter un marqueur après chaque phrase. 
*   **L'utilité** : Cela permet à l'utilisateur humain de vérifier instantanément si l'IA a bien interprété le document. C'est le contrat de confiance entre l'homme et la machine.

---

## 1. Snippets de configuration (Setup technique)
*Vous utiliserez le modèle Phi-3 chargé en Mission 2. Voici le template de prompt que vous devrez adapter.*

```python
rag_system_prompt = """You are a highly accurate Technical Support Assistant. 
Your ONLY source of truth is the provided CONTEXT below.

RULES:
1. If the answer is not in the context, strictly say: "I am sorry, but I don't have this information in my records."
2. Do NOT use your own internal knowledge. 
3. For every claim, you MUST cite the source title in brackets, like [Source: Title].
4. Keep your answer professional and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
```

---

## 2. Vos Tâches de Mission

### Tâche 1 : Conception de la Chaîne de Synthèse
Écrivez une fonction `generate_grounded_answer(query, documents)` qui :
1.  Prend la liste de documents `final_docs` (objets LangChain) issus de la Mission 2.
2.  Formate ces documents en une seule chaîne de caractères, en incluant clairement le titre de chaque document (ex: "Source: [Title] | Content: [Content]").
3.  Injecte ce contexte et la question dans le template `rag_system_prompt`.

### Tâche 2 : Inférence et Anti-Hallucination
Appelez votre modèle Phi-3 avec le prompt généré. 
**Le défi technique** : Réglez la **Température à 0.0** (Section 5.3). 

> [!IMPORTANT]
**Pourquoi ?** Nous ne voulons pas de créativité ici, nous voulons une extraction mathématique.

### Tâche 3 : Le Test de la Vérité (Citations)
Testez votre fonction avec deux cas :
1.  **Cas A** : *"What is the cost per hour for a cloud instance?"* -> La réponse doit mentionner $0.05 et citer `[Source: Public Cloud Pricing]`.
2.  **Cas B (Le Piège)** : *"Who is the current CEO of the company?"* -> Le modèle **doit** refuser de répondre.

---

## Mes avertissements

> [!WARNING]
>*   **Erreur fréquente :** Laisser le modèle s'excuser trop longuement. S'il dit "D'après les documents que vous m'avez donnés, il semble que...", il gaspille des tokens. Forcez-le à être direct.
>*   **Le paradoxe du contexte** : Si vous lui donnez 5 documents contradictoires, il risque de se perdre (*Lost in the Middle*). 

> [!TIP]
**Mon conseil :** En Mission 2, nous avons dédoublé les documents, ce qui protège votre génération actuelle !


*   **L'astuce d'ingénieur** : Si le modèle oublie de citer ses sources, ajoutez un exemple One-shot (Section 8.2) dans votre prompt système pour lui montrer le format de citation attendu. 

---

**Une fois que votre assistant est capable de répondre avec des preuves indiscutables, nous passerons à la Mission 4 : l'Audit scientifique de Fidélité avec le framework Ragas.**