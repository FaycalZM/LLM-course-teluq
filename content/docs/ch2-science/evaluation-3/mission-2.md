---
title: "Mission 2. The Bridge of Language"
weight: 2
---


# Évaluation 3 : Le Gardien de la Vérité
## Mission 2 : The Bridge of Language – Query Rewriting and Semantic Expansion

Bonjour à toutes et à tous ! J'espère que vous avez ressenti une certaine satisfaction en sécurisant votre base de données dans la Mission 1. C'est une fondation solide. Mais posons-nous une question de "réalité terrain" : vos utilisateurs sont-ils des experts en mots-clés ? 

> [!IMPORTANT]
**Je dois insister :** l'utilisateur final est souvent flou, pressé, ou ambigu. S'il demande "How much is it?", votre moteur de recherche vectoriel risque de chercher le mot "it" au lieu de comprendre qu'il parle des prix du Cloud. 

Aujourd'hui, nous allons apprendre à "traduire" l'humain pour la machine. Nous allons transformer une simple question en un faisceau de requêtes intelligentes. C'est ici que le LLM devient le majordome de votre moteur de recherche. Prêt·e·s à décupler le rappel de votre système ?


---

## Objectif de la Mission
L'étudiant doit implémenter une brique d'intelligence intermédiaire entre l'entrée utilisateur et la base FAISS. L'enjeu est de coder une chaîne de **Query Expansion (Multi-Query)**. Le LLM devra analyser la question initiale et générer trois variations sémantiques pour s'assurer que, même si l'utilisateur utilise un synonyme ou une tournure maladroite, le "Bibliothécaire" (le Retriever) ramène les documents pertinents.

---

## Les Concepts techniques : Rewriting & Multi-Query

Pour réussir cette mission, vous devez maîtriser les deux piliers de la recherche augmentée par le langage :

### 1. La Réécriture (Query Rewriting)
Comme nous l'avons vu en Section 9.2, le LLM peut servir de "nettoyeur". Si un utilisateur pose une question au sein d'une conversation (ex: "And what about the other one?"), le LLM doit être capable d'effectuer une **résolution de coréférence**. Il remplace "the other one" par le sujet réel (ex: "Internal server maintenance"). **C'est le passage de la question contextuelle à la requête autonome.** 


### 2. L'Expansion (Multi-Query RAG)
Parfois, une seule question est trop étroite. Un mot peut avoir plusieurs synonymes techniques. 
*   **Le concept** : On demande au LLM de générer 3 versions différentes de la même question.
*   **La logique** : On effectue 3 recherches dans FAISS. On fusionne les résultats. 


**Mon intuition** : C'est comme si, au lieu d'envoyer un seul chien de chasse dans la forêt, vous en envoyiez trois, chacun ayant une sensibilité légèrement différente aux odeurs. Vous avez statistiquement beaucoup plus de chances de ramener le gibier (l'information).

---

## 1. Configuration de l'environnement (Setup)
*Utilisez ces snippets pour préparer votre chaîne de réécriture. Nous utilisons ici un modèle compact pour la rapidité sur T4.*

```python
# Installation des dépendances (Vérifié pour Colab T4)
# !pip install -q transformers accelerate bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
import re

# 1. CHARGEMENT DE PHI-3-MINI (LE RÉACTEUR SÉMANTIQUE)
model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=False,
    device_map="auto"
)

# Pipeline de génération
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=False)
llm = HuggingFacePipeline(pipeline=hf_pipe)
```

---

## 2. Vos Tâches de Mission

### Tâche 1 : Le Prompt de Réécriture
Concevez un `PromptTemplate` qui force le modèle à produire exactement trois variations de la question.

> [!WARNING]
**Attention technique** : Votre prompt doit être en anglais, car votre base de connaissances technique est en anglais. Assurez-vous d'utiliser un ton professionnel.

### Tâche 2 : La Chaîne d'Expansion
Créez une fonction `expand_query(user_input)` qui :
1.  Appelle le LLM avec votre prompt d'expansion.
2.  Découpe la sortie du modèle pour obtenir une liste de 3 chaînes de caractères.
3.  **Défi technique** : Pour chaque variation, interrogez votre base FAISS (créée en Mission 1) et stockez tous les documents trouvés dans une liste unique sans doublons.

### Tâche 3 : Test de Résilience
Testez votre chaîne avec la question floue : *"How much storage do I get for free?"*.
*   Vérifiez si l'une des variations générées par l'IA contient le mot "Pricing" ou "Free tier".
*   Affichez la liste des documents récupérés. Le document "Public Cloud Pricing" doit impérativement figurer dans les résultats, même si la question originale était imprécise.

---

## Avertissements du Professeur Henni

> [!WARNING]
>*   **Erreur fréquente :** Laisser le LLM bavarder. S'il répond "Sure, here are three variations...", cela va polluer votre recherche vectorielle. Utilisez des contraintes de sortie strictes (section 8.4) pour n'avoir que les questions.
>*   **Latence :** Faire trois recherches au lieu d'une prend plus de temps. 

> **L'arbitrage de l'ingénieur** : L'expansion de requête est vitale pour le rappel (Recall), mais elle doit être utilisée avec parcimonie. Ne générez pas 50 questions !
   

**Mon conseil :** Regardez comment le modèle transforme "storage" en "disk space" ou "capacity". C'est cette richesse lexicale qui fait la force du Multi-Query RAG.

---

**Une fois que votre pont linguistique est capable de clarifier les intentions des utilisateurs, nous passerons à la Mission 3 : La Génération Ancrée avec Citations, où l'IA devra prouver ses dires.**
