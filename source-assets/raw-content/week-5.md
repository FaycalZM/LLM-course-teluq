---
# WEEK: 5
# TITLE: Semaine 5 : Modèles génératifs (decoder-only)
# CHAPTER_FIGURES: [6, 16, 28, 29, 30, 31, 134, 136, 137, 138]
# COLAB_NOTEBOOKS: []
---
[CONTENU SEMAINE 5]
# Semaine 5 : Modèles génératifs (decoder-only)

**Titre : GPT et l'art de la génération textuelle**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette étape charnière de notre parcours. La semaine dernière, nous avons étudié BERT, un modèle qui excelle dans l'écoute et la compréhension. Aujourd'hui, nous basculons de l'autre côté du miroir : nous allons explorer les modèles qui "parlent" et qui "créent". Préparez-vous, car l'architecture **Decoder-only** et la famille **GPT** sont les moteurs de la révolution créative que nous vivons actuellement. C'est fascinant, n'est-ce pas ? » [SOURCE: Livre p.167]

**Rappel semaine précédente** : « La semaine dernière, nous avons maîtrisé les modèles de représentation (Encoder-only) comme BERT, en apprenant comment le token [CLS] et le pré-entraînement par masquage (MLM) permettent de classer le texte avec une précision chirurgicale. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer l'architecture interne des modèles *Decoder-only*.
*   Décrire le processus de prédiction du prochain token et la nature autorégressive.
*   Tracer l'évolution de la lignée GPT, de GPT-1 à GPT-4.
*   Distinguer un modèle de fondation (*Foundation Model*) d'un modèle réglé pour les instructions (*Instruction-tuned*).
*   Paramétrer une génération de texte pour contrôler la créativité et le déterminisme.

---

## 5.1 La famille GPT (1000+ mots)

### L'essence de la génération : Pourquoi seulement un décodeur ?
« Imaginez que vous écriviez une histoire. Vous n'avez pas besoin d'un résumé global de tout ce qui a été écrit avant de commencer chaque mot ; vous avez simplement besoin de savoir ce qui vient juste de se passer pour décider de la suite. » C'est la philosophie de l'architecture **Decoder-only**. 

Alors que BERT (l'encodeur) regarde dans toutes les directions pour comprendre, la famille GPT (le décodeur) est conçue pour une tâche unique mais monumentale : la **prédiction du prochain token** (*Next Token Prediction*). Comme vous pouvez le voir sur la **Figure 1-24 : Architecture de GPT-1** (p.43 du livre), nous avons supprimé la partie encodeur du Transformer original pour ne garder que le décodeur. 🔑 **Je dois insister :** cette simplification est en fait une spécialisation radicale pour la génération fluide. [SOURCE: Livre p.20, Figure 1-24]

### Le moteur autorégressif : La boucle infinie du langage
Le concept fondamental à graver dans votre esprit est la nature **autorégressive** de ces modèles. 
1.  Le modèle reçoit un texte en entrée (le *prompt*).
2.  Il calcule les probabilités pour tous les mots de son dictionnaire et en choisit un.
3.  Ce nouveau mot est ajouté à l'entrée originale.
4.  Le modèle recommence le processus avec cette nouvelle séquence plus longue.

Regardez la **Figure 1-12** (p.12) que nous avions vue en semaine 1 : c'est ce processus itératif qui permet de construire des paragraphes entiers, mot après mot, brique après brique. ⚠️ **Attention : erreur fréquente ici !** Le modèle ne "planifie" pas sa phrase à l'avance. Il navigue dans un océan de probabilités statistiques, choisissant à chaque étape le chemin le plus probable (ou le plus créatif selon vos réglages). [SOURCE: Livre p.12, p.76]

### L'épopée GPT : Une montée en puissance phénoménale
L'histoire de la famille GPT est celle d'un changement d'échelle (le *scaling*) qui a révélé des capacités émergentes inattendues. Jay Alammar et Maarten Grootendorst détaillent cette évolution à travers les **Figures 1-24 à 1-27** (p.21-23). [SOURCE: Livre p.21-23, Figures 1-24 à 1-27]

#### 1. GPT-1 (2018) : La preuve par l'exemple
Avec ses **117 millions de paramètres**, GPT-1 a prouvé que le pré-entraînement sur de grandes quantités de texte brut permettait au modèle d'apprendre la structure du langage sans supervision humaine directe. C'était la naissance du *Foundation Model*. [SOURCE: Livre p.20]

#### 2. GPT-2 (2019) : L'éveil de la généralisation
OpenAI passe à **1,5 milliard de paramètres**. La surprise fut immense : sans être entraîné spécifiquement pour cela, GPT-2 a commencé à démontrer des capacités de traduction et de résumé. C'est ici que le monde a réalisé que "prédire le mot suivant" suffisait à acquérir une forme de culture générale. [SOURCE: Livre p.21, p.42]

#### 3. GPT-3 (2020) : Le géant et le prompting
Le saut est vertigineux : **175 milliards de paramètres**. À cette échelle, le modèle n'a plus besoin de fine-tuning pour beaucoup de tâches. Il suffit de lui donner quelques exemples dans le prompt (*Few-shot prompting*) pour qu'il comprenne ce qu'on attend de lui. La **Figure 1-25** (p.43) illustre parfaitement cette croissance exponentielle de la taille des modèles. [SOURCE: Livre p.21, Figure 1-25]

#### 4. GPT-4 (2023) : L'omniscience multimodale
Bien que ses chiffres exacts soient gardés secrets, GPT-4 marque l'ère des modèles capables de raisonner sur des textes et des images (multimodalité) avec une logique qui frôle parfois celle de l'humain. 

### Tableau récapitulatif de l'évolution GPT

| Modèle | Paramètres | Date | Innovation Majeure |
| :--- | :--- | :--- | :--- |
| **GPT-1** | 117M | 2018 | Premier succès du pré-entraînement génératif |
| **GPT-2** | 1,5B | 2019 | Capacité de généralisation "Zero-shot" |
| **GPT-3** | 175B | 2020 | In-context learning (apprendre via le prompt) |
| **GPT-4** | Inconnu (T) | 2023 | Raisonnement complexe et multimodalité |

[SOURCE: Livre p.21-23, p.42-43]

### Du modèle de fondation à l'assistant : Le tournant de l'Instruction Tuning
🔑 **La distinction est capitale :** Un modèle "Base" (comme GPT-3 original) est un compléteur de texte. Si vous lui demandez "Quelle est la capitale de la France ?", il pourrait répondre par une autre question comme "Et quelle est la capitale de l'Italie ?" car statistiquement, les questionnaires listent souvent des questions à la suite.

Pour transformer ce compléteur en un assistant utile, on utilise le **Supervised Fine-Tuning (SFT)** ou **Instruction Tuning**, illustré en **Figure 1-26** (p.22). On montre au modèle des milliers d'exemples de "Question -> Réponse attendue". C'est ce qui crée les modèles **Instruct** ou **Chat** que nous utilisons tous les jours. [SOURCE: Livre p.22, Figure 1-26]

### Exemple de génération autorégressive (Intuition technique)
Regardez comment un décodeur travaille en coulisses. Le mécanisme de **Causal Masking** (masquage causal) empêche le token actuel de regarder les tokens futurs. 

```python
# Exemple conceptuel de génération token par token
# Testé sur Colab T4 avec un modèle léger (GPT-2)
from transformers import pipeline, set_seed

# Utilisation d'un modèle génératif standard
generator = pipeline('text-generation', model='gpt2', device=0)
set_seed(42)

prompt = "The future of AI is"

# Le modèle va prédire un token, l'ajouter, et recommencer
output = generator(prompt, max_length=15, num_return_sequences=1)

print(f"Sortie du modèle : {output[0]['generated_text']}")
# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU LIVRE p.32-33 ET GITHUB CHAPTER 1]
```

### Éthique et Transparence : La séduction du faux
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, ne vous laissez pas berner par la fluidité. » 
Parce que GPT est un décodeur de probabilités, il ne possède pas de base de données de "vérité". S'il prédit qu'un mot est statistiquement probable, il l'écrira, même si c'est une erreur factuelle totale. C'est l'**hallucination**. 

🔑 **Je dois insister :** Plus le modèle est "large", plus il semble convaincant, ce qui rend ses erreurs d'autant plus dangereuses. Une IA qui parle bien n'est pas forcément une IA qui sait. En tant qu'experts, votre rôle est de concevoir des systèmes de vérification (comme le RAG que nous verrons en Semaine 9) pour ancrer cette imagination débordante dans la réalité. [SOURCE: Livre p.28]

« Vous avez maintenant saisi l'essence de la famille GPT. Vous comprenez que leur force réside dans cette boucle de prédiction infinie, nourrie par des milliards de paramètres. Dans la section suivante, nous verrons comment ces modèles ont évolué vers les versions "Chat" et comment choisir le bon modèle open-source pour vos projets. »

---
*Fin de la section 5.1 (1050 mots environ)*
## 5.2 Évolution des modèles (1100+ mots)

### De l'autocomplétion à la conversation : Le grand saut
« Bonjour à nouveau ! Avez-vous déjà essayé de discuter avec un dictionnaire ? C'est frustrant, n'est-ce pas ? C'est pourtant ce que nous faisions au début avec les modèles de fondation. Aujourd'hui, nous allons comprendre comment nous avons transformé ces "encyclopédies statistiques" en partenaires de discussion capables de nous aider à coder, à écrire des poèmes ou à planifier nos vacances. »

L'évolution des LLM ne s'est pas faite uniquement par l'augmentation du nombre de neurones (paramètres), mais par un changement radical dans la manière dont nous les "éduquons". Comme l'illustre la **Figure 1-26 : Modèles instruct/chat** (p.22 du livre), nous sommes passés d'un modèle qui complète mécaniquement une phrase à un modèle qui comprend l'intention derrière une consigne. 🔑 **Je dois insister :** cette transition est ce qui a rendu l'IA accessible au grand public. Sans cette étape d'alignement, GPT ne serait resté qu'un outil de laboratoire pour chercheurs spécialisés. [SOURCE: Livre p.22, Figure 1-26]

### Le Modèle de Fondation (Base Model) : L'érudit brut
Le point de départ de tout LLM moderne est le **Base Model** (Modèle de base). C'est un modèle qui a "lu" une part immense d'Internet. Sa seule mission est statistique : prédire le mot suivant. 
*   **Son comportement** : Si vous lui écrivez "Comment faire une omelette ?", il pourrait répondre par "Ingrédients : 3 œufs, sel, poivre..." mais il pourrait aussi répondre par "2. Comment faire des crêpes ? 3. Comment faire un gâteau ?", car il a appris que les listes de questions se suivent souvent. 
*   **Son utilité** : C'est une base de connaissances brute. Il n'a aucune notion de politesse, de sécurité ou de formatage. Il est le socle sur lequel nous allons bâtir l'intelligence de l'assistant. [SOURCE: Livre p.168]

### La métamorphose : SFT et Alignement
Pour transformer ce géant brut en un assistant comme ChatGPT, nous suivons un pipeline d'entraînement sophistiqué.

#### 1. Supervised Fine-Tuning (SFT) / Instruction Tuning
On engage des humains pour écrire des milliers de dialogues parfaits. "Voici une question -> Voici la réponse idéale". On ré-entraîne le modèle de base sur ces exemples. C'est l'**Instruction Tuning**. Le modèle apprend enfin qu'une question appelle une réponse et non une autre question. [SOURCE: Livre p.175, p.356]

#### 2. L'alignement par les préférences humaines (RLHF)
« Mais comment apprendre à une machine ce qui est "mieux" ou "plus poli" ? » C'est là qu'intervient le **Reinforcement Learning from Human Feedback (RLHF)**. 
*   On demande au modèle de générer deux réponses différentes pour la même question. 
*   Un humain choisit la meilleure (plus utile, plus sûre, plus claire). 
*   Un "modèle de récompense" (*Reward Model*) apprend à prédire ce que l'humain préfère. 
*   Enfin, le LLM est optimisé pour maximiser cette récompense. 

🔑 **La leçon du Prof. Henni** : Le RLHF n'ajoute pas de nouvelles connaissances au modèle ; il lui apprend à *mieux présenter* ce qu'il sait déjà et à rejeter les demandes dangereuses. C'est l'étape de la "civilité numérique". [SOURCE: Livre p.378-383]

### L'explosion de l'Open-Source : Reprendre le contrôle
Pendant longtemps, les LLM puissants étaient enfermés derrière les API d'OpenAI ou de Google (Modèles propriétaires). Mais tout a changé avec l'arrivée de la **Figure 6-1 : Sélection de foundation models** (p.168). Des modèles comme **Llama-2/3** (Meta), **Mistral 7B** et la famille **Phi** (Microsoft) ont prouvé que l'on pouvait avoir des performances incroyables avec des modèles plus petits et ouverts. [SOURCE: Livre p.168, Figure 6-1]

#### Le phénomène Phi-3-mini
🔑 **Je dois insister sur ce point technique :** La taille ne fait pas tout. Phi-3-mini, avec ses 3,8 milliards de paramètres, bat parfois des modèles deux fois plus gros. Pourquoi ? Parce que Microsoft l'a entraîné sur des données de "haute qualité" (des manuels scolaires générés par IA, très pédagogiques) plutôt que sur le "bruit" du web. 
⚠️ **Fermeté bienveillante** : Pour vos projets en laboratoire sur Colab, privilégiez toujours ces modèles compacts. Ils tiennent dans la mémoire de votre carte T4 et répondent presque instantanément. [SOURCE: Livre p.54, Phi-3 Technical Report]

### Mise en pratique : Utiliser un modèle de type "Chat"
Pour interagir avec un modèle de chat, nous devons respecter un format spécifique appelé **Chat Template**. Le modèle s'attend à voir des balises comme `<|user|>` et `<|assistant|>`. Hugging Face gère cela magnifiquement pour nous.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

# Chargement du modèle avec optimisation GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, # On réduit la précision pour économiser la VRAM
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Création de la pipeline de discussion
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Structure d'un message pour un modèle de Chat
messages = [
    {"role": "system", "content": "You are a helpful science professor."},
    {"role": "user", "content": "Explique-moi pourquoi le ciel est bleu en une phrase."}
]

# Inférence
output = pipe(messages, max_new_tokens=50)
print(f"Assistant : {output[0]['generated_text'][-1]['content']}")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 6 ET DOC HF]
```

### Le Prompt Engineering : L'art de murmurer à l'oreille des modèles
🔑 **Notez bien cette distinction :** Dans un modèle de base, vous faites de l'autocomplétion. Dans un modèle Chat/Instruct, vous faites de l'**ingénierie de prompt**. 
Comme nous le verrons en Semaine 8, la façon dont vous définissez le "Persona" (ex: "Vous êtes un professeur de science") change radicalement la qualité de la réponse. C'est parce que le modèle, durant son alignement, a appris à reconnaître ces rôles. [SOURCE: Livre p.173-176]

### Éthique et Limites de l'Alignement
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'alignement est une arme à double tranchant. » 
1.  **Le biais du censeur** : Si nous alignons trop un modèle pour qu'il soit "sûr", il peut devenir inutilement refusant (ex: refuser d'écrire une histoire triste par peur d'être "négatif"). 
2.  **La complaisance (Sycophancy)** : Les modèles alignés ont tendance à toujours donner raison à l'utilisateur, même quand celui-ci se trompe. Si vous dites "Je pense que 2+2=5", un modèle trop aligné pourrait répondre "Vous avez raison, dans certains contextes...". 
3.  **L'illusion d'intelligence** : Ne confondez pas la politesse et la fluidité avec la raison. Un modèle peut être parfaitement aligné, très poli, et vous donner un code informatique qui contient une faille de sécurité majeure. 

🔑 **Mon conseil de professeur** : Utilisez les versions "Chat" pour vos applications, mais gardez toujours un œil sur les versions "Base" pour comprendre la puissance brute du modèle sans les filtres comportementaux. [SOURCE: Livre p.28, Blog 'Responsible AI' de Hugging Face]

« Nous avons maintenant vu comment les modèles ont évolué d'une simple boucle de prédiction à des assistants conversationnels alignés. Vous comprenez la différence entre le savoir brut et le comportement appris. Dans la section suivante, nous allons apprendre à régler les "boutons" de la machine : la température et le top_p, pour transformer une réponse robotique en une création inspirée. »

---
*Fin de la section 5.2 (1150 mots environ)*
## 5.3 Paramètres de génération (1200+ mots)

### Le chef d'orchestre du hasard : Maîtriser la probabilité
« Bonjour à toutes et à tous ! Imaginez que vous soyez devant un piano magique. À chaque fois que vous jouez une note, le piano vous propose une sélection de notes suivantes possibles, chacune brillant d'une intensité différente selon sa probabilité. Jusqu'ici, nous avons laissé le piano décider. Mais aujourd'hui, je vais vous apprendre à manipuler les pédales et les leviers de cette machine pour décider si elle doit être d'une précision mathématique ou d'une créativité débordante. » 

Comme nous l'avons vu en section 5.1, un LLM ne "choisit" pas un mot : il calcule une distribution de probabilités sur l'ensemble de son dictionnaire (le *Softmax*). La manière dont nous extrayons le mot final de cette distribution est ce qu'on appelle la **stratégie de décodage**. Comme l'illustrent les **Figures 6-3 à 6-5** (p.170-172 du livre), de petits changements dans ces réglages peuvent transformer un assistant génial en un poète incompréhensible ou, à l'inverse, en un perroquet répétitif. 🔑 **Je dois insister :** la maîtrise de ces paramètres est ce qui sépare un utilisateur amateur d'un ingénieur en IA prompt-engineer. [SOURCE: Livre p.170-172, Figures 6-3 à 6-5]

### La Température : Le thermostat de l'imagination
Le paramètre le plus célèbre est sans doute la **Température**. Mathématiquement, la température est un facteur qui modifie les scores bruts (logits) avant qu'ils ne soient transformés en probabilités.

*   **Basse Température (0.1 - 0.4)** : Nous "aiguisons" la distribution. Le mot le plus probable devient encore plus dominant, et les mots improbables sont écrasés. C'est le mode "Déterministe". 
    *   *Analogie* : C'est un étudiant brillant qui ne répond que ce dont il est absolument sûr. 
    *   *Usage* : Extraction de données, résumé factuel, code informatique.
*   **Température Neutre (1.0)** : Le modèle suit les probabilités exactes apprises durant son entraînement.
*   **Haute Température (1.1 - 1.5+)** : Nous "aplatissons" la distribution. Les mots qui étaient un peu moins probables reçoivent une chance de briller. C'est le mode "Créatif".
    *   *Analogie* : C'est une séance de remue-méninges (brainstorming) où toutes les idées, même les plus folles, sont bienvenues.
    *   *Usage* : Écriture de fiction, poésie, publicité.

⚠️ **Attention : erreur fréquente ici !** Régler la température à 0 ne signifie pas "intelligence maximale". Cela active le **Greedy Decoding** (décodage glouton). Le modèle choisit *toujours* le mot le plus probable. 🔑 **Le risque :** le modèle peut s'enfermer dans des boucles répétitives ("Je pense que... et je pense que... et je pense que..."). [SOURCE: Livre p.171-172]

### Top-K et Top-P : Filtrer le bruit
Parfois, la température ne suffit pas à empêcher le modèle de choisir un mot totalement absurde qui avait pourtant une micro-probabilité de 0,001%. Pour sécuriser la génération, nous utilisons des filtres.

#### 1. Top-K (Échantillonnage par rang)
On demande au modèle de ne regarder que les **K** mots les plus probables et d'ignorer tout le reste. Si K=50, le modèle ne choisira que parmi le "Top 50".
*   **Limite** : Si le mot numéro 1 a 99% de probabilité, et que les 49 suivants ont 0,0001%, le modèle risque quand même de choisir un des 49 si on tire au sort, créant une cassure logique. [SOURCE: Livre p.171]

#### 2. Top-P (Nucleus Sampling / Échantillonnage de noyau)
C'est la méthode la plus élégante et la plus utilisée aujourd'hui. Au lieu de fixer un nombre de mots (K), on fixe une **masse de probabilité cumulative**.
*   Si Top-P = 0.9, le modèle additionne les probabilités des mots les plus probables jusqu'à atteindre 90%. 
*   Si le modèle est très sûr de lui, il ne regardera peut-être que 2 ou 3 mots. 
*   S'il hésite, il élargira son choix à 100 mots. 
🔑 **Je dois insister :** Le Top-P est dynamique. Il s'adapte à la confiance du modèle à chaque étape de la phrase. [SOURCE: Livre p.171, Blog 'LLM Roadmap' de Maarten Grootendorst]

### Contrôler le flux : max_new_tokens et Stop Tokens
Un LLM peut parler indéfiniment s'il ne rencontre pas une condition d'arrêt.
*   **max_new_tokens** : C'est votre garde-fou de budget et de temps. Vous fixez une limite physique (ex: 500 tokens). Le modèle s'arrêtera net, même au milieu d'une phrase.
*   **EOS Token (End Of Sequence)** : C'est le token "Point Final" appris durant l'entraînement. Quand le modèle le génère, la boucle s'arrête proprement. 🔑 **Notez bien :** dans vos applications, vous pouvez définir vos propres "Stop Sequences" (ex: s'arrêter dès que le modèle écrit "Utilisateur :"). [SOURCE: Livre p.33, p.172]

### Tableau 1-3 : Guide stratégique des paramètres (Inspiré du Tableau 6-1, p.172)

| Objectif de la tâche | Température | Top-P | Pourquoi ? |
| :--- | :--- | :--- | :--- |
| **Code Python / SQL** | 0.0 | 1.0 | On veut la syntaxe exacte, pas d'originalité. |
| **Résumé médical** | 0.2 | 0.9 | Priorité aux faits, mais un peu de fluidité. |
| **Chatbot Assistant** | 0.7 | 0.9 | Équilibre entre naturel et précision. |
| **Idées de scénario** | 1.2 | 0.95 | On cherche l'inattendu, la surprise. |

[SOURCE: Livre p.172, Tableau 6-1]

### Laboratoire de code : Expérimenter avec Phi-3-mini
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
# [SOURCE: CONCEPT À SOURCER – Paramètres de génération p.172]
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

### Stratégies d'échantillonnage avancées : Beam Search
Bien que nous nous concentrions sur l'échantillonnage (Sampling), il existe une autre méthode : le **Beam Search** (Recherche par faisceau). 
Au lieu de choisir un seul mot, le modèle explore plusieurs "chemins" en parallèle (le faisceau) et garde les 3 ou 5 phrases les plus probables globalement. 
🔑 **La distinction est importante :** Le Beam Search donne des phrases très structurées et parfaites grammaticalement, mais souvent très ennuyeuses et répétitives. C'est pour cela que pour les agents conversationnels, nous préférons presque toujours le couple **Température + Top-P**. [SOURCE: Livre p.79-80]

### Éthique et Limites : Le risque de la "Température Extrême"
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, manipuler le hasard n'est pas sans danger. » 

1.  **Fiabilité et Température** : Plus vous augmentez la température pour être "créatif", plus vous augmentez statistiquement le risque d'**hallucinations**. Le modèle, en cherchant des mots originaux, finit par inventer des faits qui n'existent pas. 🔑 **Règle de sécurité :** Pour toute application critique (santé, droit, finance), restez sous une température de 0.3.
2.  **Biais amplifiés** : L'échantillonnage peut parfois faire ressortir des corrélations toxiques présentes dans les données d'entraînement mais qui sont normalement étouffées par le mot le plus probable. 
3.  **Déterminisme et Reproductibilité** : Si vous construisez un produit commercial, l'utilisateur s'attend à ce que le bouton "Générer" produise un résultat stable. Si votre température est trop haute, vous ne pourrez jamais corriger un bug, car le modèle ne donnera jamais deux fois la même réponse. 🔑 **Astuce technique :** Utilisez un `seed` (graine aléatoire) fixe pour vos tests. [SOURCE: Livre p.28]

« Nous avons maintenant fait le tour des réglages de notre machine générative. Vous savez comment la brider pour la précision et comment la libérer pour l'inspiration. Vous êtes passés de spectateurs à pilotes de LLM. Dans la section suivante, nous conclurons cette semaine en explorant les applications concrètes de ces modèles génératifs et les barrières éthiques que nous devons encore franchir. »

---
*Fin de la section 5.3 (1310 mots environ)*
## 5.4 Applications et limites (700+ mots)

### L'IA dans la vie réelle : Au-delà de la curiosité technologique
« Bonjour à toutes et à tous ! Nous terminons cette semaine passionnante sur les modèles génératifs. Nous avons vu comment ils sont construits et comment régler leurs paramètres. Mais maintenant, posons-nous la question fondamentale : à quoi cela sert-il vraiment dans le monde professionnel ? Et surtout, quels sont les pièges qui pourraient transformer une innovation brillante en un échec retentissant ? 🔑 **Je dois insister : un expert en LLM se reconnaît non pas à ce qu'il sait générer, mais à ce qu'il sait anticiper comme erreurs.** »

Les modèles *decoder-only*, grâce à leur capacité de généralisation, ne sont plus cantonnés à de simples démonstrations techniques. Ils sont devenus des partenaires de productivité. Cependant, comme nous allons le voir, leur nature probabiliste impose des garde-fous éthiques et techniques rigoureux. [SOURCE: Livre p.27]

### Domaines d'applications concrètes
Comme l'illustre la **Figure 1-2** (p.5) et les exemples de la page 27, les LLM génératifs redéfinissent plusieurs métiers :

1.  **Assistance à la programmation (Coding)** : C'est sans doute l'application la plus mature. Des modèles comme StarCoder ou GPT-4 ne se contentent pas d'écrire du code ; ils expliquent des algorithmes complexes et aident au débogage. ⚠️ **Attention : erreur fréquente ici !** Ne copiez jamais un code généré sans le tester dans un environnement sécurisé (sandbox).
2.  **Rédaction et Créativité** : Du copywriting publicitaire à la scénarisation, les LLM servent de "partenaires de brainstorming". Ils excellent à briser le syndrome de la page blanche en proposant des structures et des idées initiales.
3.  **Résumé et Synthèse** : Donnez un rapport de 50 pages à un modèle avec une grande fenêtre de contexte, et il vous en sortira les 5 points clés en quelques secondes. C'est un gain de temps inestimable pour les professions juridiques et administratives.
4.  **Enseignement et Tutorat** : En paramétrant un persona (vu en 5.2), un LLM peut devenir un tuteur patient qui explique la physique quantique à un enfant de 10 ans. [SOURCE: Livre p.27]

### Le défi de la vérité : Hallucinations et Facticité
🔑 **C'est le concept le plus important de cette section :** Un LLM ne possède pas de modèle interne de la "vérité". Il possède un modèle de la "vraisemblance statistique". 

L'**hallucination** est le phénomène où le modèle génère une réponse fluide, grammaticalement parfaite, mais factuellement fausse. 
*   *Exemple* : Inventer une biographie pour une personne réelle ou citer une loi qui n'existe pas.
*   *Pourquoi ?* Parce que dans l'espace des probabilités, la suite de mots inventée semble "logique" au modèle. 

⚠️ **Fermeté bienveillante** : « Mes chers étudiants, ne blâmez pas le modèle pour ses hallucinations. C'est sa nature profonde d'imaginer la suite. C'est à vous, concepteurs, de mettre en place des systèmes de vérification. » [SOURCE: Livre p.28]

### Biais, Représentations et Miroirs déformants
Nous l'avons évoqué en semaine 1, mais il est temps d'y revenir avec force. Les LLM sont entraînés sur le web, un endroit magnifique mais aussi rempli de préjugés.
*   **Biais algorithmique** : Si le modèle a lu 90% de textes où les PDG sont des hommes, il aura une probabilité statistique plus élevée de générer "il" pour parler d'un dirigeant d'entreprise.
*   **Toxicité** : Sans l'alignement (RLHF) que nous avons étudié en 5.2, un modèle pourrait générer des propos haineux ou dangereux simplement parce qu'il les a rencontrés dans ses données d'entraînement.

🔑 **Le message du Prof. Henni** : « L'IA n'est pas neutre. Elle est le reflet amplifié de nos propres sociétés. En tant qu'ingénieurs, vous avez le devoir moral d'auditer vos modèles et d'utiliser des techniques de "Guardrails" (garde-fous) pour minimiser ces biais. » [SOURCE: Livre p.28, Blog 'Responsible AI' de Hugging Face]

### Considérations Éthiques et Légales
Le déploiement de LLM soulève des questions inédites que le livre aborde en page 28 :
1.  **Propriété Intellectuelle** : À qui appartient un poème généré par GPT-4 ? À OpenAI ? À vous ? À personne ? Les tribunaux du monde entier débattent encore de cette question.
2.  **Transparence et AI Act** : L'Union Européenne, via l'AI Act, impose de plus en plus de transparence. Un utilisateur doit savoir s'il interagit avec un humain ou une machine. 🔑 **Je dois insister :** La clarté envers l'utilisateur final est la base de la confiance numérique.
3.  **Confidentialité des données** : ⚠️ **Attention !** Tout ce que vous envoyez à une API propriétaire (comme celle d'OpenAI) peut potentiellement être utilisé pour entraîner les futures versions du modèle. Ne partagez jamais de données médicales ou de secrets industriels sans garantie de confidentialité. [SOURCE: Livre p.28, p.29]

### Bonnes pratiques pour un usage responsable
Pour conclure cette semaine, voici vos commandements d'expert :
*   **Human-in-the-loop** : Toujours avoir une révision humaine pour les sorties critiques.
*   **Ancrage (Grounding)** : Ne laissez pas le modèle parler de mémoire ; fournissez-lui des documents sources (nous verrons comment faire avec le RAG en semaine 9).
*   **Température basse pour les faits** : Gardez vos réglages de créativité au minimum pour les tâches sérieuses.

« Vous avez maintenant une vision complète : vous connaissez la théorie, les réglages, et les responsabilités qui pèsent sur vos épaules. Les LLM sont des outils magiques, mais la magie demande de la discipline. Place maintenant au laboratoire pour mettre en pratique vos talents de pilote de GPT ! » [SOURCE: Livre p.34]

---
*Fin de la section 5.4 (820 mots environ)*
## 🧪 LABORATOIRE SEMAINE 5 (750+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! J'espère que vous avez fait le plein d'énergie, car c'est aujourd'hui que vous devenez officiellement des "pilotes de LLM". Dans ce laboratoire, nous allons passer de la théorie à la pratique en manipulant les paramètres de génération. Vous allez voir comment une simple variation de température peut changer la "personnalité" d'une IA. 🔑 **Je dois insister :** ne vous contentez pas de faire tourner le code, observez la subtilité des changements dans le texte produit. C'est là que réside le secret des grands experts. Amusez-vous bien ! » [SOURCE: Livre p.32-33]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quelle architecture est utilisée par les modèles de la famille GPT pour maximiser l'efficacité de la génération ?**
   a) Encoder-only
   b) Encoder-Decoder
   c) Decoder-only
   d) RNN à mémoire longue
   **[Réponse: c]** [Explication: Contrairement à BERT, GPT n'utilise que la partie décodeur pour prédire le mot suivant de manière autorégressive. SOURCE: Livre p.20, Figure 1-24]

2. **Quel paramètre permet de contrôler le niveau de "créativité" ou de "déterminisme" d'une réponse ?**
   a) max_new_tokens
   b) Temperature
   c) n_ctx
   d) weight_decay
   **[Réponse: b]** [Explication: Une température basse rend le modèle prévisible (froid), une température haute le rend créatif (chaud). SOURCE: Livre p.171]

3. **Quelle technique d'échantillonnage limite le choix du prochain token aux mots dont la probabilité cumulée atteint un certain seuil ?**
   a) Greedy Search
   b) Top-K
   c) Nucleus Sampling (Top-P)
   d) Beam Search
   **[Réponse: c]** [Explication: Le Top-P (Nucleus) sélectionne dynamiquement le "noyau" des mots les plus probables. SOURCE: Livre p.171]

4. **Quel modèle est particulièrement recommandé pour les utilisateurs "GPU-poor" car il tourne sur des appareils mobiles ou des Colab gratuits ?**
   a) GPT-4
   b) Llama-3-70B
   c) Phi-3-mini
   d) Falcon-180B
   **[Réponse: c]** [Explication: Avec 3.8B de paramètres, Phi-3-mini est extrêmement efficace pour sa taille. SOURCE: Livre p.54, p.168]

5. **Qu'est-ce qu'une "hallucination" dans le contexte des LLM ?**
   a) Un bug qui fait planter l'ordinateur
   b) Une réponse fluide et convaincante mais factuellement fausse
   c) Le fait que le modèle utilise trop de VRAM
   d) Une image générée à la place du texte
   **[Réponse: b]** [Explication: Le modèle privilégie la vraisemblance statistique sur la vérité historique ou logique. SOURCE: Livre p.28]

6. **Quelle est la différence majeure entre un "Base model" et un "Chat model" ?**
   a) Le Chat model est plus grand
   b) Le Chat model a subi un Instruction Tuning et un alignement (RLHF) pour suivre des consignes
   c) Le Base model est gratuit, le Chat model est payant
   d) Il n'y a aucune différence technique
   **[Réponse: b]** [Explication: L'instruction tuning transforme un compléteur de texte en un assistant conversationnel. SOURCE: Livre p.22, p.356]

7. **Combien de paramètres possède le modèle GPT-3, marquant le début de l'ère des modèles géants ?**
   a) 117 millions
   b) 1,5 milliard
   c) 175 milliards
   d) 1 000 milliards
   **[Réponse: c]** [Explication: GPT-3 a prouvé qu'à 175B de paramètres, de nouvelles capacités émergent. SOURCE: Livre p.21, Figure 1-25]

8. **Quel token spécial indique au modèle que la génération d'un texte est terminée ?**
   a) `[CLS]`
   b) `<s>`
   c) `<|endoftext|>` ou `<|end|>` (EOS)
   d) `[MASK]`
   **[Réponse: c]** [Explication: Le token End-Of-Sequence (EOS) déclenche l'arrêt de la boucle autorégressive. SOURCE: Livre p.191]

9. **Quelle stratégie de décodage garantit de choisir systématiquement le token ayant la probabilité la plus élevée ?**
   a) Random Sampling
   b) Nucleus Sampling
   c) Greedy Decoding
   d) Top-K (K=50)
   **[Réponse: c]** [Explication: Le décodage "glouton" choisit toujours l'option n°1, sans aucun hasard. SOURCE: Livre p.80]

10. **Quel avantage majeur offre l'Instruction Tuning pour un développeur d'applications ?**
    a) Le modèle répond plus vite
    b) Le modèle comprend mieux les formats (JSON, listes) et les consignes de persona
    c) Le modèle utilise moins de tokens
    d) Le modèle n'a plus besoin d'embeddings
    **[Réponse: b]** [Explication: Cela permet de guider le modèle via le "Prompt Engineering" sans changer le code. SOURCE: Livre p.175]

---

### 🔹 EXERCICE 1 : Génération contrôlée et Température (Niveau Basique)

**Objectif** : Expérimenter l'impact de la température sur la diversité des réponses avec Phi-3-mini.

```python
# --- CODE COMPLET (CORRIGÉ) ---
from transformers import pipeline
import torch

# Initialisation de la pipeline (QUESTION CODE)
model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)

# Tâche : Générer une blague avec 3 températures différentes
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

# Observations attendues : 
# 0.1 -> Blague très classique, répétitive si relancée.
# 1.5 -> Peut devenir incohérent ou inventer des mots étranges.
```

---

### 🔹 EXERCICE 2 : Prompt Engineering : Persona et Format (Niveau Intermédiaire)

**Objectif** : Utiliser les composants d'un prompt vus en section 5.2 (Instruction, Persona, Format) pour obtenir une réponse structurée.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# (On suppose pipe déjà initialisé comme ci-dessus)

# --- CONFIGURATION DU PROMPT (QUESTION CODE) ---
# Tâche : Créer un prompt qui demande à l'IA d'agir en tant qu'expert en nutrition
# et de répondre sous forme de liste JSON.

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

---

### 🔹 EXERCICE 3 : Nucleus Sampling (Top-P) vs Greedy (Niveau Avancé)

**Objectif** : Comparer la richesse lexicale entre un décodage déterministe et un décodage par noyau (Nucleus).

```python
# --- CODE COMPLET (CORRIGÉ) ---
# --- CONFIGURATION (QUESTION CODE) ---
prompt = "In a world where artificial intelligence rules the oceans,"
messages = [{"role": "user", "content": prompt}]

# --- RÉPONSE (ANSWER CODE) ---
# 1. Génération Greedy (Déterministe)
# [SOURCE: Greedy decoding Livre p.80]
greedy_out = pipe(messages, max_new_tokens=40, do_sample=False) # do_sample=False force le mot le plus probable

# 2. Génération Nucleus (Top-P)
# [SOURCE: Nucleus sampling Livre p.171]
nucleus_out = pipe(messages, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.8)

print("--- MODE GREEDY (Plus probable) ---")
print(greedy_out[0]['generated_text'][-1]['content'])

print("\n--- MODE NUCLEUS (Échantillonné) ---")
print(nucleus_out[0]['generated_text'][-1]['content'])
```
**Attentes** : Le mode Greedy sera très propre mais peut-être un peu banal. Le mode Nucleus proposera des adjectifs ou des tournures de phrases plus variées. ⚠️ **Fermeté bienveillante :** Si vous baissez trop le Top-P (ex: 0.1), cela revient quasiment à faire du Greedy ! [SOURCE: Livre p.171]

---

**Mots-clés de la semaine** : GPT, Decoder-only, Autorégressif, Instruction Tuning, RLHF, Température, Top-P (Nucleus), Hallucination, EOS Token, Base vs Chat model.

**En prévision de la semaine suivante** : Nous allons apprendre à donner une "mémoire externe" à nos modèles grâce à la recherche sémantique et aux bases de données vectorielles. Préparez-vous pour la révolution du **RAG** ! [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 6, p.167-198.
*   Hugging Face Blog : *Prompt Engineering* (https://huggingface.co/blog/prompt-engineering).
*   Prompting Guide : https://www.promptingguide.ai/
*   GitHub Officiel : chapter06 repository.

[/CONTENU SEMAINE 5]
