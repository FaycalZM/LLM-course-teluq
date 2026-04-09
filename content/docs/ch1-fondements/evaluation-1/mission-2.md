---
title: "Mission 2. Analyse de l'Anisotropie"
weight: 2
---
{{< katex />}}

# Évaluation 1 – Mission 2 : Analyse de l'Anisotropie (Le Phénomène du Cône)
## Audit de la distribution géométrique des vecteurs de la couche finale de GPT-2

Bonjour à toutes et à tous ! Dans cette mission, nous allons analyser un phénomène fascinant appelé l'**anisotropie** (ou l'effet de "cône").

> [!IMPORTANT]
**Note technique :** 
L'anisotropie est considérée comme la "maladie génétique" des Transformers. Si les vecteurs ne sont pas distribués de manière uniforme, la similarité cosinus devient un indicateur biaisé. Nous devons quantifier ce biais pour évaluer la capacité du modèle à distinguer le signal du bruit.

---

## 1. Préparation de l'environnement
*Temps estimé : 2 minutes. Assurez-vous d'être en mode "GPU T4" (Exécution > Modifier le type d'exécution).*

```python
# Installation des dépendances
# !pip install -q transformers torch numpy scipy matplotlib seaborn scikit-learn

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Vérification du GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du matériel : {device}")

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True).to(device)
model.eval()
```

---

## 2. Implémentation technique
Nous allons extraire des vecteurs pour 500 tokens aléatoires à la couche 0 et à la couche 12, puis comparer leurs distributions de similarité.

```python
# Sélection de 500 tokens aléatoires (excluant les tokens de contrôle)
vocab_size = tokenizer.vocab_size
random_token_ids = random.sample(range(100, vocab_size - 100), 500)

def get_embeddings_at_layer(token_ids, layer_idx):
    """
    Extrait les vecteurs pour une liste d'IDs à une couche spécifique.
    """
    ids_tensor = torch.tensor([token_ids]).to(device) 
    with torch.no_grad():
        outputs = model(ids_tensor)
    
    selected_layer = outputs.hidden_states[layer_idx] 
    return selected_layer[0].cpu().numpy()

# --- CONFIGURATION DE LA MISSION (QUESTION CODE) ---
# TÂCHE : Extraire les embeddings aux couches 0 et 12, puis calculer la similarité cosinus moyenne.

# 1. Récupérez les embeddings pour embeddings_layer_0 et embeddings_layer_12
# ...

def compute_avg_cosine_sim(embeddings):
    # 2. Implémentez le calcul de la similarité cosinus moyenne en ignorant la diagonale
    # ...
    pass

# 3. Affichez et comparez les résultats !
```

---

## 3. Analyse et Justifications (À remettre par l'étudiant)

**Résultats attendus :**
Vous devriez observer que la similarité cosinus moyenne est basse (entre 0.1 et 0.3) à la **couche 0 (Embedding structuré)**, tandis qu'à la **couche 12**, elle est très élevée (souvent > 0.85 pour GPT-2). C'est la signature de l'**anisotropie**. Les vecteurs sont tous "poussés" dans la même direction.

> [!NOTE]
> * La solution complète avec le code matplotlib/seaborn pour générer les graphiques de densité KDE et l'analyse approfondie est incluse dans le notebook Colab qui sera partagé avec le prof.

---

## Question Critique de Mission
**Question** : Quelle est la cause structurelle de cette dégénérescence des représentations à la couche 12 de GPT-2, et quel est son impact concret sur la recherche sémantique ?

**Réponse attendue** : Ce phénomène est dû à la dominance de certains composants (poids et biais) dans les couches de normalisation et d'attention qui forcent les vecteurs vers une direction commune pour faciliter la tâche de prédiction du prochain token (Softmax). Avec une similarité moyenne élevée, un vecteur reste difficile à distinguer. Cela dégrade fortement les résultats d'une recherche sémantique, car le modèle peine à percevoir les nuances. 

---

> [!WARNING]
**Note d'Éthique Algorithmique** : L'anisotropie engendre des biais de groupement. Les mots liés à des groupes minoritaires ou sujets moins fréquents peuvent être "éjectés" du cône géométrique, constituant une forme de discrimination. Gardez cela en tête !
