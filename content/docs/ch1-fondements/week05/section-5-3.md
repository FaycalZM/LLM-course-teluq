---
title: "5.3 Paramètres de génération"
weight: 4
---

## Le chef d'orchestre du hasard : Maîtriser la probabilité
Bonjour à toutes et à tous ! Imaginez que vous soyez devant un piano magique. À chaque fois que vous jouez une note, le piano vous propose une sélection de notes suivantes possibles, chacune brillant d'une intensité différente selon sa probabilité. Jusqu'ici, nous avons laissé le piano décider. Mais aujourd'hui, je vais vous apprendre à manipuler les pédales et les leviers de cette machine pour décider si elle doit être d'une précision mathématique ou d'une créativité débordante.

Comme nous l'avons vu en [**section 5.1**]({{< relref "section-5-1.md" >}}), un LLM ne "choisit" pas un mot : il calcule une distribution de probabilités sur l'ensemble de son dictionnaire (le *Softmax*). La manière dont nous extrayons le mot final de cette distribution est ce qu'on appelle la **stratégie de décodage**. 

{{< bookfig src="136.png" week="05" >}}

Comme l'illustrent les **Figures 5-6 à 5-8**, de petits changements dans ces réglages peuvent transformer un assistant génial en un poète incompréhensible ou, à l'inverse, en un perroquet répétitif. 

> [!IMPORTANT]
🔑 **Je dois insister :** la maîtrise de ces paramètres est ce qui sépare un utilisateur amateur d'un ingénieur en IA prompt-engineer.

## La Température : Le thermostat de l'imagination
Le paramètre le plus célèbre est sans doute la **Température**. Mathématiquement, la température est un facteur qui modifie les scores bruts (logits) avant qu'ils ne soient transformés en probabilités.

*   **Basse Température (0.1 - 0.4)** : Nous "aiguisons" la distribution. Le mot le plus probable devient encore plus dominant, et les mots improbables sont écrasés. C'est le mode "Déterministe". 
    *   *Analogie* : C'est un étudiant brillant qui ne répond que ce dont il est absolument sûr. 
    *   *Usage* : Extraction de données, résumé factuel, code informatique.
*   **Température Neutre (1.0)** : Le modèle suit les probabilités exactes apprises durant son entraînement.
*   **Haute Température (1.1 - 1.5+)** : Nous "aplatissons" la distribution. Les mots qui étaient un peu moins probables reçoivent une chance de briller. C'est le mode "Créatif".
    *   *Analogie* : C'est une séance de remue-méninges (brainstorming) où toutes les idées, même les plus folles, sont bienvenues.
    *   *Usage* : Écriture de fiction, poésie, publicité.

<a id="fig-5-7"></a>

{{< bookfig src="137.png" week="05" >}}

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Régler la température à 0 ne signifie pas "intelligence maximale". Cela active le **Greedy Decoding** (décodage glouton). Le modèle choisit *toujours* le mot le plus probable.

> [!CAUTION]
🚩 **Le risque :** le modèle peut s'enfermer dans des boucles répétitives ("Je pense que... et je pense que... et je pense que...").

## Top-K et Top-P : Filtrer le bruit
Parfois, la température ne suffit pas à empêcher le modèle de choisir un mot totalement absurde qui avait pourtant une micro-probabilité de 0,001%. Pour sécuriser la génération, nous utilisons des filtres.

### 1. Top-K (Échantillonnage par rang)
On demande au modèle de ne regarder que les **K** mots les plus probables et d'ignorer tout le reste. Si K=50, le modèle ne choisira que parmi le "Top 50".
*   **Limite** : Si le mot numéro 1 a 99% de probabilité, et que les 49 suivants ont 0,0001%, le modèle risque quand même de choisir un des 49 si on tire au sort, créant une cassure logique.

### 2. Top-P (Nucleus Sampling / Échantillonnage de noyau)
C'est la méthode la plus élégante et la plus utilisée aujourd'hui. Au lieu de fixer un nombre de mots (K), on fixe une **masse de probabilité cumulative**.
*   Si Top-P = 0.9, le modèle additionne les probabilités des mots les plus probables jusqu'à atteindre 90%. 
*   Si le modèle est très sûr de lui, il ne regardera peut-être que 2 ou 3 mots. 
*   S'il hésite, il élargira son choix à 100 mots.

{{< bookfig src="138.png" week="05" >}}

> [!IMPORTANT]
🔑 **Je dois insister :** Le Top-P est dynamique. Il s'adapte à la confiance du modèle à chaque étape de la phrase.


## Contrôler le flux : max_new_tokens et Stop Tokens
Un LLM peut parler indéfiniment s'il ne rencontre pas une condition d'arrêt.
*   **max_new_tokens** : C'est votre garde-fou de budget et de temps. Vous fixez une limite physique (ex: 500 tokens). Le modèle s'arrêtera net, même au milieu d'une phrase.
*   **EOS Token (End Of Sequence)** : C'est le token "Point Final" appris durant l'entraînement. Quand le modèle le génère, la boucle s'arrête proprement. 

> [!NOTE]
🔑 **Notez bien :** dans vos applications, vous pouvez définir vos propres "Stop Sequences" (ex: s'arrêter dès que le modèle écrit "Utilisateur :").

## Guide stratégique des paramètres 

| Objectif de la tâche | Température | Top-P | Pourquoi ? |
| :--- | :--- | :--- | :--- |
| **Code Python / SQL** | 0.0 | 1.0 | On veut la syntaxe exacte, pas d'originalité. |
| **Résumé médical** | 0.2 | 0.9 | Priorité aux faits, mais un peu de fluidité. |
| **Chatbot Assistant** | 0.7 | 0.9 | Équilibre entre naturel et précision. |
| **Idées de scénario** | 1.2 | 0.95 | On cherche l'inattendu, la surprise. |


## Laboratoire de code : Expérimenter avec Phi-3-mini
Voici comment implémenter ces paramètres avec la bibliothèque Transformers. Je vous encourage vivement à tester ce code sur Colab et à changer les valeurs pour voir l'assistant "changer de personnalité".

```python
# Testé pour Google Colab T4 16GB VRAM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

messages = [{"role": "user", "content": "Raconte-moi une histoire très courte sur un robot qui découvre une fleur."}]

# --- CONFIGURATION DE GÉNÉRATION ---
output = pipe(
    messages,
    max_new_tokens=100,
    do_sample=True,      # Activer l'échantillonnage (nécessaire pour temp/top_p)
    temperature=0.8,     # Créativité modérée
    top_p=0.9,           # Nucleus sampling pour la cohérence
    repetition_penalty=1.2 # Éviter les boucles de phrases identiques
)

print(output[0]['generated_text'][-1]['content'])
```

## Stratégies d'échantillonnage avancées : Beam Search
Bien que nous nous concentrions sur l'échantillonnage (Sampling), il existe une autre méthode : le **Beam Search** (Recherche par faisceau). Au lieu de choisir un seul mot, le modèle explore plusieurs "chemins" en parallèle (le faisceau) et garde les 3 ou 5 phrases les plus probables globalement. 

> [!IMPORTANT]
🔑 **La distinction est importante :** Le Beam Search donne des phrases très structurées et parfaites grammaticalement, mais souvent très ennuyeuses et répétitives. C'est pour cela que pour les agents conversationnels, nous préférons presque toujours le couple **Température + Top-P**.

## Éthique et Limites : Le risque de la "Température Extrême"

> [!WARNING]
⚠️ Mes chers étudiants, manipuler le hasard n'est pas sans danger. 

>1.  **Fiabilité et Température** : Plus vous augmentez la température pour être "créatif", plus vous augmentez statistiquement le risque d'**hallucinations**. Le modèle, en cherchant des mots originaux, finit par inventer des faits qui n'existent pas. 
>> [!TIP]
🔑 **Règle de sécurité :** Pour toute application critique (santé, droit, finance), restez sous une température de 0.3.

>2.  **Biais amplifiés** : L'échantillonnage peut parfois faire ressortir des corrélations toxiques présentes dans les données d'entraînement mais qui sont normalement étouffées par le mot le plus probable. 
>3.  **Déterminisme et Reproductibilité** : Si vous construisez un produit commercial, l'utilisateur s'attend à ce que le bouton "Générer" produise un résultat stable. Si votre température est trop haute, vous ne pourrez jamais corriger un bug, car le modèle ne donnera jamais deux fois la même réponse. 

> [!TIP]
🔑 **Astuce technique :** Utilisez un `seed` (graine aléatoire) fixe pour vos tests -> assurer la "reproductibilité".

---
Nous avons maintenant fait le tour des réglages de notre machine générative. Vous savez comment la brider pour la précision et comment la libérer pour l'inspiration. Vous êtes passés de spectateurs à pilotes de LLM. Dans la section suivante, nous conclurons cette semaine en explorant les applications concrètes de ces modèles génératifs et les barrières éthiques que nous devons encore franchir.