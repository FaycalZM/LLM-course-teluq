---
title: "Examen Final "
weight: 7
---

# L'Examen Final

Bonjour à toutes et à tous. Nous y sommes. C'est le moment de vérité, mais ne le voyez pas comme une épreuve de peur, voyez-le comme votre rapport final d'expertise. Vous avez les outils, vous avez l'intuition, et vous avez la rigueur. 


> [!IMPORTANT]
**Je dois insister :** je ne cherche pas des perroquets savants capables de recracher des définitions. Je cherche des ingénieurs capables de justifier un choix technique sous la contrainte. 

Lisez chaque question avec attention, visualisez le flux des données, et rappelez-vous que derrière chaque équation, il y a un objectif d'utilité et d'éthique. Bonne chance à tous, je sais que vous êtes prêts !

## 🔹 PARTIE 1 : Questions de Diagnostic et de Conception

```json
{
  "exam_title": "SCI2070 Final Exam - Part 1: Advanced Concepts & Justifications",
  "total_questions": 15,
  "instructions": "Répondez de manière précise. Pour les questions demandant une justification, développez votre raisonnement technique en vous appuyant sur les concepts vus en cours.",
  "questions": [
    {
      "id": 1,
      "topic": "Architecture Transformer",
      "question": "Pourquoi le facteur d'échelle (1/sqrt(dk)) est-il mathématiquement indispensable dans le calcul de l'attention de Vaswani et al. ?",
      "expected_answer": "Il est indispensable pour stabiliser les gradients. Sans lui, pour de grandes dimensions de vecteurs (dk), le produit scalaire Q.K peut atteindre des valeurs très élevées, ce qui pousse la fonction Softmax vers des zones où le gradient est quasi nul (saturation). Cela empêcherait le modèle d'apprendre efficacement pendant la backpropagation."
    },
    {
      "id": 2,
      "topic": "Inférence et Optimisation",
      "question": "Expliquez le compromis (trade-off) imposé par l'utilisation du KV Cache en production.",
      "expected_answer": "Le compromis se situe entre le temps de calcul (latence) et l'utilisation de la mémoire VRAM. Le KV Cache réduit la latence en évitant de recalculer l'attention pour les jetons passés à chaque étape de génération (complexité linéaire vs quadratique). Cependant, il consomme une quantité massive de VRAM pour stocker les vecteurs Keys et Values, ce qui limite le nombre d'utilisateurs simultanés (batch size) que le GPU peut traiter."
    },
    {
      "id": 3,
      "topic": "Fine-tuning (PEFT)",
      "question": "Dans la méthode LoRA, quel est l'impact d'un choix de rang (r) trop élevé par rapport à un rang faible ?",
      "expected_answer": "Un rang (r) plus élevé augmente la capacité de mémorisation et la finesse d'adaptation du modèle, mais il augmente aussi proportionnellement le nombre de paramètres entraînables, la consommation de VRAM et le risque de sur-apprentissage (overfitting) sur le dataset de fine-tuning."
    },
    {
      "id": 4,
      "topic": "Recherche Sémantique",
      "question": "Pourquoi un Bi-Encoder est-il utilisé pour la recherche initiale (Retrieval) tandis qu'un Cross-Encoder est réservé au Reranking ?",
      "expected_answer": "C'est une question de complexité computationnelle. Le Bi-Encoder permet de pré-calculer les embeddings et d'utiliser des index comme FAISS pour fouiller des millions de documents en millisecondes. Le Cross-Encoder est beaucoup plus précis car il permet l'attention totale entre la question et le document, mais sa complexité est trop lourde pour être appliquée à l'intégralité d'une base de données en temps réel."
    },
    {
      "id": 5,
      "topic": "Alignement (DPO)",
      "question": "Quelle est la différence fondamentale dans l'utilisation du 'modèle de référence' entre le RLHF et le DPO ?",
      "expected_answer": "En RLHF, le modèle de référence sert de contrainte via la divergence KL pour éviter que le modèle ne diverge trop du SFT. En DPO, le modèle de référence est utilisé directement dans la fonction de perte pour calculer le ratio de probabilité entre le modèle actuel et le modèle initial, éliminant ainsi le besoin d'un modèle de récompense (Reward Model) explicite."
    },
    {
      "id": 6,
      "topic": "Tokenisation",
      "question": "Quelles sont les conséquences techniques d'un 'Fragmentation Ratio' élevé pour une langue spécifique ?",
      "expected_answer": "Un ratio élevé signifie qu'un mot est découpé en trop de jetons. Cela entraîne : 1. Une saturation précoce de la fenêtre de contexte (on peut lire moins de texte). 2. Une augmentation du coût de calcul et de la latence. 3. Une dégradation potentielle de la sémantique, car le modèle doit faire plus d'efforts pour lier des morceaux de mots fragmentés."
    },
    {
      "id": 7,
      "topic": "Multimodalité",
      "question": "Dans l'architecture BLIP-2, quel est le rôle précis du Q-Former ?",
      "expected_answer": "Le Q-Former agit comme un pont sémantique. Il utilise des 'queries' apprenables pour interroger l'encodeur visuel (ViT) et en extraire uniquement les caractéristiques pertinentes pour le langage. Il réduit ainsi la complexité visuelle en un nombre fixe de tokens digestes pour le LLM."
    },
    {
      "id": 8,
      "topic": "Ingénierie des Prompts",
      "question": "Qu'est-ce que le phénomène 'Lost in the Middle' et comment influence-t-il la conception d'un RAG ?",
      "expected_answer": "Les modèles ont tendance à mieux retenir les informations situées au début (Primacy) et à la fin (Recency) d'un prompt long. En RAG, cela signifie que si vous injectez 20 documents, les plus pertinents doivent être placés aux extrémités du prompt augmenté pour éviter que le modèle ne les ignore."
    },
    {
      "id": 9,
      "topic": "RAG (Evaluation)",
      "question": "Si un système RAG obtient un score de Faithfulness (Fidélité) de 1.0 mais un score de Answer Relevancy de 0.1, quel est le défaut majeur du système ?",
      "expected_answer": "Le système ne ment pas (tout ce qu'il dit est dans le texte), mais il est hors-sujet. Il n'a pas compris l'intention de l'utilisateur et renvoie probablement des informations extraites du contexte qui ne répondent pas à la question posée."
    },
    {
      "id": 10,
      "topic": "Quantification",
      "question": "Pourquoi le type de données NF4 (Normal Float 4) est-il supérieur au FP4 standard pour la quantification des poids ?",
      "expected_answer": "Parce que les poids des réseaux de neurones suivent généralement une distribution normale (en cloche). Le NF4 est conçu pour que chaque intervalle de quantification contienne un nombre égal de poids, minimisant ainsi l'erreur de reconstruction par rapport à un espacement linéaire uniforme (FP4)."
    },
    {
      "id": 11,
      "topic": "Agents et Autonomie",
      "question": "Justifiez l'importance de l'étape 'Thought' dans le cycle ReAct (Thought-Action-Observation).",
      "expected_answer": "L'étape 'Thought' active le raisonnement de Système 2. Elle permet au modèle de planifier ses actions et de justifier pourquoi il choisit tel outil. Sans cela, le modèle agirait par réflexe statistique (Système 1), ce qui augmente les risques d'erreurs d'outils et de boucles infinies."
    },
    {
      "id": 12,
      "topic": "Éthique et Biais",
      "question": "Comment la similarité cosinus peut-elle involontairement amplifier des biais lors d'une recherche de CV par embeddings ?",
      "expected_answer": "Si les embeddings ont appris des corrélations sexistes (ex: 'Ingénieur' proche de 'Homme'), la similarité cosinus va mathématiquement favoriser les profils masculins pour un poste technique, même si le genre n'est pas mentionné, en s'appuyant sur des 'proxys' sémantiques présents dans les descriptions."
    },
    {
      "id": 13,
      "topic": "Sécurité",
      "question": "Décrivez le mécanisme d'une 'Indirect Prompt Injection'.",
      "expected_answer": "L'injection indirecte se produit lorsque l'attaquant cache des instructions malveillantes dans une source externe (ex: un site web) que le LLM va lire via un outil (RAG ou recherche web). Le modèle ingère ces instructions comme s'il s'agissait de données neutres et finit par exécuter l'ordre caché."
    },
    {
      "id": 14,
      "topic": "Architecture (Normalisation)",
      "question": "Quel est l'avantage de la 'Pre-Normalization' par rapport à la 'Post-Normalization' dans l'entraînement de modèles profonds ?",
      "expected_answer": "La Pre-Norm normalise les activations avant chaque bloc d'attention ou MLP. Cela crée un 'chemin direct' (identity path) plus stable pour les gradients via les connexions résiduelles, permettant d'entraîner des modèles beaucoup plus profonds sans divergence catastrophique."
    },
    {
      "id": 15,
      "topic": "Futur des LLM",
      "question": "En quoi les State Space Models (SSM) comme Mamba diffèrent-ils fondamentalement des Transformers en termes de mémoire ?",
      "expected_answer": "Les Transformers gardent tout le passé en mémoire (KV Cache), ce qui est lourd. Les SSM compressent tout le passé dans un 'état interne' de taille fixe. Leur mémoire ne grandit pas avec la longueur du texte, ce qui permet une complexité linéaire au lieu de quadratique."
    }
  ]
}
```

---

## 🔹 PARTIE 2 : Exercices de Mise en Pratique (Code & Analyse)

### Exercice 1 : Diagnostic de l'Anisotropie

**Objectif** : Mesurer si un modèle "souffre" d'un regroupement excessif de ses vecteurs.

```python
# --- CODE À COMPLÉTER ---
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# Tâche : Calculez la similarité moyenne entre 2 mots totalement opposés : "war" et "peace"
word1 = "war"
word2 = "peace"

# 1. Obtenez les embeddings de la DERNIÈRE couche pour ces deux mots
# [VOTRE CODE ICI]

# --- SOLUTION ATTENDUE ---
# [SOURCE: Évaluation 1 Concept]
def get_emb(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[0, -1, :].numpy()

vec1 = get_emb(word1)
vec2 = get_emb(word2)
sim = cosine_similarity([vec1], [vec2])[0][0]

print(f"Similarité entre '{word1}' et '{word2}' : {sim:.4f}")

# QUESTION : Si le résultat est de 0.85, le modèle est-il isotrope ou anisotrope ? Justifiez.
# RÉPONSE : Anisotrope. 0.85 est une similarité très élevée pour des antonymes. 
# Cela prouve l'effet de cône (Section 2.4).
```

#### Résultats Attendus pour l'Évaluation :

1. **Valeur numérique** : La similarité cosinus obtenue pour GPT-2 entre "war" et "peace" à la couche 12 doit se situer entre **0.80 et 0.95**.
    
2. **Analyse attendue** : L'étudiant doit noter que ce score est anormalement élevé pour des contraires.
    
3. **Justification technique** : Le modèle souffre d'**anisotropie**. Les couches finales compressent tous les vecteurs dans une direction commune pour faciliter la prédiction du prochain token (Softmax), ce qui fausse la mesure de distance sémantique.


### Exercice 2 : RAG avec Filtrage par Métadonnées

**Objectif** : Implémenter un moteur de recherche qui respecte les contraintes de sécurité.

```python
# --- CODE À COMPLÉTER ---
# On utilise LangChain et FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Tâche : Créez une base de données avec 2 documents et cherchez uniquement les docs 'Public'
# [VOTRE CODE ICI]

# --- SOLUTION ATTENDUE ---
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content="Secret formula is 123", metadata={"status": "Private"}),
    Document(page_content="The weather is nice", metadata={"status": "Public"})
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(docs, embeddings)

# Effectuez la recherche avec un filtre
results = vector_db.similarity_search("formula", k=1, filter={"status": "Public"})

# QUESTION : Pourquoi n'avons-nous pas trouvé la "Secret formula" ?
# RÉPONSE : Car le filtre de métadonnées a exclu le document avant le calcul de similarité.
```

#### Résultats Attendus pour l'Évaluation :

1. **Sortie du code** : L'objet retourné doit être soit une liste vide [], soit le document sur "Paris", mais en aucun cas le document contenant le "secret code".
    
2. **Analyse attendue** : L'étudiant doit confirmer que le document confidentiel est devenu **invisible** pour le moteur de recherche.
    
3. **Justification technique** : Le **Pre-filtering** sur les métadonnées intervient avant le calcul de similarité, ce qui est la méthode la plus sûre pour prévenir les fuites de données sémantiques vers le LLM.
---

### Exercice 3 : Configuration de Fine-tuning LoRA

**Objectif** : Paramétrer correctement un adaptateur pour un GPU T4.

```python
# --- CODE À COMPLÉTER ---
from peft import LoraConfig

# Tâche : Configurez un adaptateur LoRA avec un rang de 16, un alpha de 32 
# ciblant les couches Query et Value.

# --- SOLUTION ATTENDUE ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# QUESTION : Pourquoi utiliser lora_alpha = 2 * r ?
# RÉPONSE : C'est une règle empirique qui stabilise l'apprentissage en équilibrant 
# le poids de l'adaptateur par rapport au modèle original.
```

#### Résultats Attendus pour l'Évaluation :

1. **Validité de l'objet** : L'objet peft_config doit avoir r=8, lora_alpha=16 et target_modules=["q_proj", "v_proj"].
    
2. **Analyse attendue** : L'étudiant doit expliquer que ce réglage permet de n'entraîner qu'une fraction infime (souvent < 0.1%) des paramètres du modèle.
    
3. **Justification technique** : Le choix de alpha = 2 * r assure une mise à l'échelle stable des poids appris, évitant que l'adaptateur ne déstabilise les connaissances pré-entraînées du modèle de base.
---

### Exercice 4 : Inférence avec Contrainte de Format

**Objectif** : Forcer une sortie JSON valide en utilisant les paramètres de génération.

```python
# --- CODE À COMPLÉTER ---
# Tâche : Complétez l'appel pour obtenir une réponse déterministe (Temp = 0)
# et limitez la réponse à 50 nouveaux tokens.

# output = model.generate(
#    input_ids,
#    ... [VOTRE CODE ICI] ...
# )

# --- SOLUTION ATTENDUE ---
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=False,  # Force la température à 0 (implicite)
    temperature=1.0,  # Ignoré si do_sample=False
    top_p=1.0         # Ignoré
)

# QUESTION : Quel est l'avantage du do_sample=False pour une sortie JSON ?
# RÉPONSE : La reproductibilité. On veut que la syntaxe JSON soit identique 
# à chaque appel pour ne pas briser l'intégration logicielle.
```

#### Résultats Attendus pour l'Évaluation :

1. **Paramètres clés** : Le code doit impérativement contenir do_sample=False (ou temperature=0.0) et max_new_tokens=30.
    
2. **Analyse attendue** : L'étudiant doit mentionner que ce mode est le **Greedy Decoding**.
    
3. **Justification technique** : Pour générer du JSON ou du code, le déterminisme est vital. Le do_sample=False garantit que pour une même entrée, le modèle produira toujours la même sortie, ce qui facilite les tests unitaires et l'intégration dans des pipelines logiciels de production.
---

**Grille de notation** :

- **Code fonctionnel** : 50% de la note de l'exercice.
    
- **Justification théorique correcte** : 30% de la note.
    
- **Référence aux limites (VRAM, Biais, Latence)** : 20% de la note.


> [!NOTE]
**Ma note finale** :  
Vous avez terminé votre examen. Je vous rappelle que la note finale n'est qu'un chiffre, mais la rigueur que vous avez montrée aujourd'hui est une compétence pour la vie. Prenez le temps de relire vos justifications, assurez-vous que vos sources sont cohérentes. Je vous souhaite une excellente carrière dans le monde passionnant des LLM. Au plaisir de vous croiser en tant que collègues !

--- 
**"N'oubliez pas : un bon ingénieur sait expliquer son code à son manager autant qu'à son compilateur. Bonne chance !"**

***- Professeur Khadidja Henni -***