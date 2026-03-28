---
title: "12.3 DPO (Direct Preference Optimization)"
weight: 4
---

{{< katex />}}

## La révolution de la simplicité : L'IA qui n'a plus besoin de carotte
Bonjour à toutes et à tous ! J'espère que vous avez les yeux bien ouverts, car nous allons aujourd'hui aborder ce qui a été, pour beaucoup d'entre nous dans la communauté de recherche, un véritable choc intellectuel. 

Dans la section précédente (12.2), je vous ai montré la complexité monumentale du RLHF : trois modèles différents, un algorithme PPO capricieux et instable, et des semaines de réglages. 

> [!IMPORTANT]
‼️ **Je dois insister :** pendant deux ans, nous pensions que c'était le seul moyen.

Puis, en 2023, des chercheurs de Stanford (Rafailov et al., 2023) ont publié l'algorithme **DPO** (*Direct Preference Optimization*). Ils ont prouvé que nous pouvions obtenir le même alignement, voire meilleur, sans aucun apprentissage par renforcement. C'est l'application parfaite du rasoir d'Ockham à l'IA. 

Respirez, nous allons voir comment transformer un casse-tête de renforcement en un simple problème de classification.

---
## L'intuition mathématique : Pourquoi faire compliqué quand on peut faire direct ?
Pour comprendre DPO, nous devons d'abord comprendre pourquoi le RLHF est si lourd. Dans le RLHF, nous entraînons un modèle de récompense pour qu'il devienne une "carotte" que le LLM essaie d'attraper. Mais comme je vous l'ai dit, cette carotte est une approximation. 

> [!TIP]
💡 **Le coup de génie de DPO :** Les auteurs ont réalisé qu'il existe une relation mathématique directe entre la récompense optimale et la probabilité des mots générés par le modèle. 

> En d'autres termes : si un modèle préfère générer la réponse A plutôt que la réponse B, c'est *comme s'il* s'attribuait lui-même une récompense. Au lieu d'entraîner un juge (Reward Model) puis un élève (LLM), pourquoi ne pas utiliser directement les probabilités du modèle pour l'aligner ? 

Comme l'illustre la **Figure 12-11 : Le LLM comme son propre modèle de récompense** , nous supprimons totalement le modèle de récompense séparé.

{{< bookfig src="288.png" week="12" >}}

**Analyse détaillée de la Figure 12-11** :
*   **Le modèle Trainable (au centre)** : C'est le modèle que nous sommes en train d'éduquer.
*   **Le modèle Reference (en haut)** : C'est une copie "gelée" de notre modèle SFT initial (celui de la Semaine 11). 
*   **Le mécanisme de comparaison** : La figure montre que pour un même prompt, on présente au modèle une réponse "Acceptée" et une réponse "Rejetée". On regarde comment le modèle trainable se comporte par rapport au modèle de référence. 

> [!TIP]
💭 **Mon intuition :** DPO ne demande pas au modèle d'être "bon" dans l'absolu. Il lui demande d'augmenter l'écart de probabilité entre ce que l'humain aime et ce qu'il n'aime pas, tout en restant fidèle à ses connaissances de base.

---
## Analyse technique : La danse des log-probabilités

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent que DPO est juste un autre type de fine-tuning supervisé. Ce n'est pas le cas. Le SFT cherche à maximiser la probabilité d'une réponse. Le DPO cherche à maximiser la **préférence relative**.

Regardons la **Figure 12-12 : Calcul du shift dans les scores de rejet** . C'est l'illustration la plus "mathématique" de cette semaine.

{{< bookfig src="289.png" week="12" >}}

**Explication de la Figure 12-12** :
Cette figure décompose le calcul de la "perte" (loss) DPO. Pour chaque mot (token) d'une réponse :
1.  On calcule la probabilité que le modèle trainable lui attribue.
2.  On calcule la probabilité que le modèle de référence (le modèle d'origine) lui attribuait.
3.  On fait le ratio de ces deux probabilités (en passant par les logarithmes, ce qu'on appelle les **Log-Probabilités**). 
4.  **Le verdict** : Si le ratio augmente pour la réponse "Choisie" (Accepted) et diminue pour la réponse "Rejetée", alors le modèle apprend correctement. 

> [!IMPORTANT]
📌 **Je dois insister sur ce point :** DPO utilise le modèle de référence comme une "ancre". 

> Sans cette ancre, le modèle pourrait devenir très poli mais perdre totalement sa capacité à parler français ou anglais correctement (il s'égarerait dans des suites de mots absurdes mais plaisantes). La figure montre bien ce "shift" (décalage) que l'algorithme essaie d'optimiser. [SOURCE: Livre p.385]

---

## Le paramètre Beta ($\beta$) : Le thermostat de l'alignement
L'un des réglages les plus importants du DPO (et de la classe `DPOConfig` que nous verrons en code) est le paramètre **Beta**. 

🛠️ **Sa fonction** : Il contrôle la force de la "laisse" entre le modèle trainable et le modèle de référence. 
> *   **Si Beta est petit (ex: 0.01)** : On laisse au modèle beaucoup de liberté pour s'aligner sur les préférences humaines. Il deviendra très obéissant, mais risque de perdre sa fluidité d'origine ou d'halluciner.
> *   **Si Beta est grand (ex: 0.5)** : On force le modèle à rester très proche de son état initial. L'alignement sera plus subtil, mais la langue restera très naturelle.

> [!TIP]
👉🏻 **Mon conseil** : Dans vos projets sur Colab, une valeur de **0.1** est souvent le "point magique" qui offre un bon compromis entre sécurité et intelligence.

---
## Pourquoi DPO a-t-il "tué" le RLHF pour beaucoup d'entre nous ?
Si vous travaillez dans une startup ou un laboratoire de recherche, vous choisirez DPO 9 fois sur 10. Voici pourquoi :
1.  **Stabilité numérique** : PPO (RLHF) est connu pour ses "explosions de gradient" où le modèle devient soudainement idiot. DPO est une simple descente de gradient classique, très stable.
2.  **Consommation de ressources** : En RLHF, vous devez charger le LLM, le Reward Model et le modèle de référence en même temps. En DPO, vous n'avez besoin que du LLM et de la référence (qui peut souvent être déchargée ou quantifiée). 
3.  **Vitesse** : Pas besoin de phase d'échantillonnage complexe pendant l'entraînement. DPO est environ 3 à 5 fois plus rapide à entraîner que le RLHF.

---
## Mise en œuvre pratique : Le DPOTrainer
Pour implémenter cela, nous utilisons la bibliothèque **TRL** (*Transformer Reinforcement Learning*). Contrairement au **SFTTrainer** de la semaine dernière, le `DPOTrainer` attend un format de données très spécifique : des triplets composés du prompt, de la réponse choisie et de la réponse rejetée. 

Le workflow commence par un **Templating** des données d'alignement. 

> [!NOTE]
✍🏻 **Je dois insister :** si votre template de prompt (les balises `<|user|>`, etc.) n'est pas identique à celui utilisé pendant le SFT (Semaine 11), le DPO va échouer lamentablement. Le modèle sera perdu entre deux formats différents.

### Laboratoire de code : Configuration DPO sur Colab
Voici comment configurer un entraînement DPO moderne. Nous utilisons la quantification pour que tout rentre dans les 16 Go de votre carte T4.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install trl transformers peft accelerate bitsandbytes

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. CHARGEMENT DU MODÈLE AVEC QUANTIFICATION (Section 11.3)
# On utilise le modèle que nous avons fine-tuné en SFT
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. LES DEUX MODÈLES : TRAINABLE ET REFERENCE
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
# En mode PEFT/LoRA, on n'a pas besoin de charger une 2ème copie physique
# Le DPOTrainer gère la référence via les poids gelés
ref_model = None 

# 3. CONFIGURATION DES ARGUMENTS
dpo_config = DPOConfig(
    output_dir="./tinyllama_dpo",
    beta=0.1,                # Le thermostat de l'alignement
    learning_rate=5e-7,      # Très faible ! On ne veut pas casser le modèle
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=512,
    max_prompt_length=256,
    optim="paged_adamw_32bit"
)

# 4. INITIALISATION DU TRAINER
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=my_preference_dataset, # Contient 'prompt', 'chosen', 'rejected'
    tokenizer=tokenizer,
    peft_config=peft_config # On continue d'utiliser LoRA !
)

print("DPO prêt à l'emploi. Le modèle va apprendre à préférer l'humain !")
```


‼️ Notez le `learning_rate` extrêmement bas (`5e-7`). 

> **C'est une règle de survie :** en alignement, on ne cherche plus à apprendre de nouvelles choses au modèle, on cherche à "incliner" légèrement ses préférences. Si vous mettez un taux trop fort, vous allez détruire des mois de pré-entraînement en quelques minutes.

---
## Éthique et Responsabilité : Les dangers de l'"Alignement de Façade"

> [!CAUTION]
⚖️ Mes chers étudiants, l'alignement par DPO est une chirurgie esthétique du comportement. 

1.  **L'émergence des "Reward Hackers"** : Même avec DPO, le modèle peut trouver des raccourcis. S'il remarque que les réponses "humaines" commencent toujours par "En tant qu'intelligence artificielle...", il pourrait commencer à ajouter cette phrase partout sans devenir plus utile pour autant. 
2.  **L'uniformité culturelle** : DPO va forcer le modèle à converger vers les préférences de vos annotateurs. Si vos données de préférence viennent uniquement d'un groupe social restreint, vous êtes en train de "tuer" la diversité sémantique du modèle au nom de la sécurité. 
>> [!TIP]
👉🏻 **Mon conseil de professeur** : Utilisez des datasets de préférence diversifiés (comme *Intel Orca DPO*) qui mélangent raisonnement logique et sécurité.
3.  **L'oubli des minorités** : Un modèle trop aligné sur la "majorité" peut devenir incapable de comprendre ou de répondre correctement à des sous-cultures ou des langues régionales qu'il connaissait pourtant bien après le pré-entraînement.

---
## Synthèse : DPO vs PPO
| Dimension | PPO (RLHF Classique) | DPO (Direct Optimization) |
| :--- | :--- | :--- |
| **Complexité** | Élevée (3 modèles + boucle RL) | Faible (1 modèle + 1 référence) |
| **Stabilité** | Capricieux, instable | Très stable (Classification simple) |
| **Mémoire VRAM** | Massive | Modérée (compatible LoRA/QLoRA) |
| **Nécessité de RM** | Oui, doit être entraîné d'abord | Non, le LLM est son propre RM |

> [!TIP]
✉️ **Mon message final pour cette section** : Le DPO est la preuve que nous comprenons de mieux en mieux la "physique" interne des modèles de langage.
 
> Nous n'avons plus besoin de systèmes de récompense complexes pour parler à l'IA ; nous lui parlons directement via sa propre structure de probabilités. C'est un pas immense vers une IA plus prévisible et plus sûre. Mais n'oubliez pas : une IA "bien élevée" n'est pas forcément une IA qui a raison. Gardez toujours votre esprit critique.

---
Nous avons terminé notre plongée dans la technique de pointe de l'alignement. Vous savez désormais comment éduquer un modèle sans les lourdeurs du renforcement classique. Dans la dernière section de cette semaine ➡️, nous verrons comment juger si tout ce travail a porté ses fruits : nous parlerons d'évaluation humaine, de Chatbot Arena et de la difficulté de mesurer la "sagesse" d'une machine.