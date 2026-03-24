---
title: "Laboratoire "
weight: 6
---

Bonjour à toutes et à tous ! J'espère que vous avez fait le plein d'énergie, car c'est aujourd'hui que vous devenez officiellement des "pilotes de LLM". Dans ce laboratoire, nous allons passer de la théorie à la pratique en manipulant les paramètres de génération. Vous allez voir comment une simple variation de température peut changer la "personnalité" d'une IA. 

> [!IMPORTANT]
🔑 **Je dois insister :** ne vous contentez pas de faire tourner le code, observez la subtilité des changements dans le texte produit. C'est là que réside le secret des grands experts. Amusez-vous bien !

---

## 🔹 EXERCICE 1 : Génération contrôlée et Température (Niveau Basique)

**Objectif** : Expérimenter l'impact de la température sur la diversité des réponses avec Phi-3-mini.

```python
from transformers import pipeline
import torch

# Initialisation de la pipeline (QUESTION CODE)
model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

#VOTRE TÂCHE : Générer une blague avec 3 températures différentes (par exemple: 0.1, 0.7, 1.5)
```
<details>
<summary>Voir la réponse</summary>

```python
prompt = "Tell me a very short joke about a computer."
messages = [{"role": "user", "content": prompt}]

# --- RÉPONSE (ANSWER CODE) ---
temperatures = [0.1, 0.7, 1.5]

for temp in temperatures:
    # On utilise do_sample=True pour permettre l'usage de la température
    # [SOURCE: Paramètres de génération Livre p.172]
    output = pipe(messages, max_new_tokens=30, do_sample=True, temperature=temp)
    print(f"\n--- Température: {temp} ---")
    print(output[0]['generated_text'][-1]['content'])
```

**Observations attendues**: 
*   0.1 -> Blague très classique, répétitive si relancée.
*   0.7 -> Blague plus créative, moins prévisible.
*   1.5 -> Peut devenir incohérent ou inventer des mots étranges.


</details>

---

## 🔹 EXERCICE 2 : Prompt Engineering : Persona et Format (Niveau Intermédiaire)

**Objectif** : Utiliser les composants d'un prompt vus en section 5.2 (Instruction, Persona, Format) pour obtenir une réponse structurée.

```python
# (On suppose pipe déjà initialisé comme ci-dessus)

# --- CONFIGURATION DU PROMPT (QUESTION CODE) ---
# Tâche : Créer un prompt qui demande à l'IA d'agir en tant qu'expert en nutrition
# et de répondre sous forme de liste JSON.

```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Anatomie d'un prompt Livre p.173-178]
system_msg = "You are a professional nutritionist. Always respond in JSON format."
user_msg = "Give me 3 healthy breakfast ideas with their main ingredient."

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg}
]

# Inférence avec température basse pour respecter le format JSON
output = pipe(messages, max_new_tokens=150, temperature=0.1, do_sample=True)

print(output[0]['generated_text'][-1]['content'])

```
</details>

---

## 🔹 EXERCICE 3 : Nucleus Sampling (Top-P) vs Greedy (Niveau Avancé)

**Objectif** : Comparer la richesse lexicale entre un décodage déterministe et un décodage par noyau (Nucleus).

```python
# --- CONFIGURATION (QUESTION CODE) ---
prompt = "In a world where artificial intelligence rules the oceans,"
messages = [{"role": "user", "content": prompt}]
```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (ANSWER CODE) ---
# 1. Génération Greedy (Déterministe)
greedy_out = pipe(messages, max_new_tokens=40, do_sample=False) # do_sample=False force le mot le plus probable

# 2. Génération Nucleus (Top-P)
# [SOURCE: Nucleus sampling Livre p.171]
nucleus_out = pipe(messages, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.8)

print("--- MODE GREEDY (Plus probable) ---")
print(greedy_out[0]['generated_text'][-1]['content'])

print("\n--- MODE NUCLEUS (Échantillonné) ---")
print(nucleus_out[0]['generated_text'][-1]['content'])
```
**Attentes** : Le mode Greedy sera très propre mais peut-être un peu banal. Le mode Nucleus proposera des adjectifs ou des tournures de phrases plus variées. 

> [!NOTE]
⚠️ Si vous baissez trop le Top-P (ex: 0.1), cela revient quasiment à faire du Greedy !

</details>

---

**Mots-clés de la semaine** : GPT, Decoder-only, Autorégressif, Instruction Tuning, RLHF, Température, Top-P (Nucleus), Hallucination, EOS Token, Base vs Chat model.

**En prévision de la semaine suivante** : Nous allons apprendre à donner une "mémoire externe" à nos modèles grâce à la recherche sémantique et aux bases de données vectorielles. Préparez-vous pour la révolution du **RAG** !

**SOURCES COMPLÈTES** :
*   Hugging Face Blog : *Prompt Engineering* (https://huggingface.co/blog/prompt-engineering).
*   Prompting Guide : https://www.promptingguide.ai/
*   GitHub Officiel : chapter06 repository.