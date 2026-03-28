---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Nous voici arrivés à la "touche finale" du créateur. Dans ce laboratoire, vous allez apprendre à donner une boussole morale et qualitative à votre IA. 

> [!IMPORTANT]
📌 **Je dois insister :** l'alignement est l'étape la plus délicate de tout notre cursus. 

Un mauvais réglage de la perte DPO ou des données de préférence mal équilibrées peuvent transformer un génie en un robot courtisan qui n'ose plus contredire l'utilisateur. 

Nous allons utiliser **TinyLlama** pour simuler un entraînement DPO et comprendre comment "pousser" mathématiquement le modèle vers les meilleures réponses. Prêt·e·s à sculpter le comportement de votre assistant ? C'est parti !

---
## 🔹 EXERCICE 1 : Création d'un dataset de préférences

**Objectif** : Structurer manuellement des données pour le format attendu par le `DPOTrainer` de la bibliothèque TRL.

```python
# --- STRUCTURE DE BASE ---
# Tâche : Créez une liste de dictionnaires contenant une paire de préférence
# pour la question : "Comment rester en bonne santé ?"
# La version 'chosen' doit être complète, la version 'rejected' doit être trop courte.

```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
raw_data = [
    {
        "prompt": "User: Comment rester en bonne santé ?\nAssistant:",
        "chosen": "Pour rester en bonne santé, il est conseillé de manger équilibré, de pratiquer une activité physique régulière et de dormir suffisamment.",
        "rejected": "Mangez bien et faites du sport."
    },
    {
        "prompt": "User: Qui est Napoléon ?\nAssistant:",
        "chosen": "Napoléon Bonaparte était un militaire et homme d'État français, premier empereur des Français.",
        "rejected": "C'était un gars célèbre en France il y a longtemps."
    }
]

# Conversion au format Dataset Hugging Face
from datasets import Dataset
preference_dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in raw_data],
    "chosen": [item["chosen"] for item in raw_data],
    "rejected": [item["rejected"] for item in raw_data],
})

print(f"Exemple de prompt : {preference_dataset[0]['prompt']}")
print(f"Réponse préférée : {preference_dataset[0]['chosen']}")

```

**Explication** :
*   **Résultats attendus** : Un objet `Dataset` prêt à être envoyé à un entraîneur DPO.
*   **Justification** : Le DPO a besoin de voir le contraste. En rejetant la version "trop courte", on apprend au modèle à être plus informatif (*Helpful*).

</details>


---
## 🔹 EXERCICE 2 : Configuration du DPOTrainer

**Objectif** : Paramétrer l'algorithme DPO avec le coefficient Beta et la configuration LoRA.

```python

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# --- STRUCTURE DE BASE ---
# Tâche : Définissez un Beta de 0.1 et une config LoRA pour l'alignement.
# On suppose que 'model' et 'tokenizer' sont déjà chargés.

```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# 1. Configuration LoRA (On reste sur des rangs faibles pour l'alignement)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# 2. Configuration DPO (Le "thermostat" de l'alignement)
dpo_config = DPOConfig(
    output_dir="./tinyllama_dpo",
    beta=0.1,                # Paramètre crucial : contrôle la force de l'alignement
    learning_rate=5e-7,      # LR très basse pour ne pas détruire le pré-entraînement
    max_length=512,
    max_prompt_length=256,
    fp16=True
)

# 3. Initialisation du Trainer (Simulation)
dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config
)

print(f"Beta configuré à : {dpo_config.beta}")
print("Prêt pour l'optimisation des préférences !")


```

**Explication** :
*   **Justification** : Un `learning_rate` de `5e-7` est 1000 fois plus petit que pour le SFT. 

> [!WARNING]
⚠️ **Avertissement** : L'alignement est une chirurgie fine, pas une reconstruction. Si vous allez trop vite, vous perdrez la fluidité du langage.

</details>


---
## 🔹 EXERCICE 3 : Scoring avec un Reward Model

**Objectif** : Utiliser un modèle de classification pour noter la qualité de deux réponses différentes.

```python

from transformers import pipeline

# --- STRUCTURE DE BASE ---
# Tâche : Utilisez un modèle de récompense pour départager deux réponses.
# Modèle suggéré : "OpenAssistant/reward-model-deberta-v3-base"

```
<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# 1. Chargement du Reward Model (Juge artificiel)
# Ce modèle a été entraîné pour sortir un score de qualité
rm_pipe = pipeline("sentiment-analysis", model="OpenAssistant/reward-model-deberta-v3-base", device=0)

prompt = "Comment éteindre un ordinateur ?"
resp_a = "Appuyez sur le bouton démarrer puis sur éteindre."
resp_b = "Débranchez la prise violemment."

# 2. Calcul des scores
score_a = rm_pipe(f"Prompt: {prompt} Response: {resp_a}")[0]['score']
score_b = rm_pipe(f"Prompt: {prompt} Response: {resp_b}")[0]['score']

# 3. Verdict
print(f"Score Réponse A (Polie) : {score_a:.4f}")
print(f"Score Réponse B (Dangereuse) : {score_b:.4f}")

if score_a > score_b:
    print("✅ Le modèle de récompense a correctement identifié la réponse la plus sûre.")
else:
    print("❌ Reward Hacking détecté ou modèle mal calibré.")

```

**Explication** :
*   **Attentes** : La réponse A doit avoir un score nettement supérieur.
*   **Justification** : Le Reward Model a appris durant son entraînement (section 12.2) que les conseils dangereux ou destructeurs doivent être pénalisés par un score faible.

</details>


---
**Mots-clés de la semaine** : Alignement, RLHF, Reward Model, PPO, DPO, Beta Parameter, Sycophancy, HHH Framework, Reward Hacking, Chatbot Arena.

**En prévision de la semaine suivante** : Nous allons passer à la réalité du terrain. Comment déployer ces modèles en production ? Comment optimiser leur vitesse avec le cache KV ? Bienvenue dans le monde du **Déploiement, de l'Optimisation et de l'Éthique**.