---
title: "Mission 1. L'Évolution Sémantique Inter-couches"
weight: 1
---
{{< katex />}}

# Évaluation 1 – Mission 1 : L'Évolution Sémantique Inter-couches
## Analyse de la polysémie d'un mot à travers les 12 blocs du Transformer

Bonjour à toutes et à tous ! Dans cette mission, nous allons "écouter" les conversations privées entre les couches du modèle. 

> [!IMPORTANT]
**Je dois insister :** ne regardez pas seulement le résultat final. Observez comment, couche après couche, la machine affine sa compréhension. 

Au début, elle ne voit que des lettres et de la syntaxe. À la fin, elle saisit le concept. C'est ce voyage sémantique que nous allons mesurer.

---

## 1. Préparation de l'environnement
*Temps estimé : 2 minutes. Assurez-vous d'être en mode "GPU T4" (Exécution > Modifier le type d'exécution).*

```python
# Installation des dépendances
# !pip install -q transformers torch matplotlib seaborn scikit-learn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Vérification du GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du matériel : {device}")

```

---

## 2. Implémentation technique
Nous allons comparer le mot **"avocat"** dans deux contextes radicalement différents.

```python
model_id = "gpt2" # Modèle Decoder-only avec 12 couches
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True).to(device)

# --- CONFIGURATION DE LA MISSION (QUESTION CODE) ---
# Phrase A (Weight) : "This bag is very light and easy to carry." 
# Phrase B (Brightness) : "Please turn on the light because it's dark." 
# Note : Pour garder la polysémie pure en français, nous utiliserions un modèle comme CamemBERT. 
# Ici, nous utilisons GPT-2 avec deux phrases illustrant le concept de contexte.

sentence_1 = "This bag is very light and easy to carry."
sentence_2 = "Please turn on the light because it's dark." # On force l'usage du même mot pour tester le contexte - vous pouvez utiliser d'autres phrases pour tester

word_to_test = "light

# TÂCHE : Extraire les hidden states et comparer les vecteurs du mot cible à chaque couche.
```

---

## 3. Analyse et Justifications (À remettre par l'étudiant)

**Résultats attendus :**
Vous devriez observer que la similarité cosinus est très élevée (proche de 0.9 ou 1.0) à la **couche 0 (Embedding)**, puis qu'elle **diminue progressivement** à mesure que l'on monte dans les couches supérieures.

> [!NOTE]
> * La solution complète (code + analyse et justification) est incluse dans le notebook Colab qui sera partagé avec le prof.

---

## Question Critique de Mission
**Question** : Pourquoi la similarité ne tombe-t-elle jamais à 0, même à la couche 12 ?

**Réponse attendue** : Parce que le modèle conserve une partie de l'identité lexicale du mot (sa forme orthographique) et parce que les embeddings de GPT-2 souffrent d'**anisotropie** (ils sont tous un peu regroupés dans la même direction de l'espace). 

---

> [!TIP]
**Mon conseil** : Gardez ce script précieusement, il est la base de l'audit de "compréhension" de n'importe quel modèle que vous rencontrerez en entreprise.
