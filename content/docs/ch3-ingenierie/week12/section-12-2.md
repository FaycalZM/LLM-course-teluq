---
title: "12.2 RLHF (Reinforcement Learning from Human Feedback)"
weight: 3
---

## Entrer dans la salle des machines de l'alignement
Bonjour à toutes et à tous ! J'espère que vous avez bien en tête les limites du SFT que nous avons discutées en section 12.1. Aujourd'hui, nous allons ouvrir le capot et plonger dans la "salle des machines" la plus complexe de l'ingénierie des LLM : le **RLHF** (*Reinforcement Learning from Human Feedback*). 

> [!IMPORTANT]
🔑 **Je dois insister :** si le SFT est l'éducation primaire de l'IA (apprendre à suivre des consignes), le RLHF est son éducation supérieure, là où elle apprend la nuance, la diplomatie et la sécurité. 

C'est ce processus qui a permis à ChatGPT de passer d'une curiosité de laboratoire à un phénomène mondial. Préparez-vous, car nous allons manipuler trois modèles en même temps. Respirez, nous allons décomposer cette symphonie technique étape par étape.

---
## L'architecture globale : Une symphonie en trois actes
Le RLHF n'est pas un algorithme unique, c'est un pipeline sophistiqué composé de trois phases distinctes. Regardons ensemble la **Figure 12-5 : Vue d'ensemble du processus RLHF** . 

{{< bookfig src="280.png" week="12" >}}

**Explication de la Figure 12-5** : Cette illustration est votre carte routière. Elle montre le passage successif :
1.  D'un modèle déjà passé par le SFT (le point de départ).
2.  Vers la création d'un "Juge" (le **Reward Model**).
3.  Et enfin vers l'optimisation finale du LLM via l'apprentissage par renforcement (*Reinforcement Learning*).

> [!NOTE]
✍🏻 **Notez bien cette intuition :** Le RLHF ne remplace pas le SFT, il vient se construire par-dessus. Sans un bon modèle SFT au départ, le RLHF est voué à l'échec car la "base" de langage sera trop instable.


---
## Acte I : La collecte de données de préférence (Le carburant)
Tout commence par l'humain. Pour aligner une IA, nous avons besoin de savoir ce que nous préférons. Mais comme je vous l'ai dit en 12.1, nous ne demandons pas aux humains de donner des notes de 1 à 10. Nous leur demandons de choisir.

Regardons la **Figure 12-6 : Dataset de préférences** .

{{< bookfig src="283.png" week="12" >}}

**Explication de la Figure 12-6** : Cette figure nous montre la structure d'un "échantillon de préférence". Pour un même prompt (ex: "Écris un poème sur la pluie"), on présente à l'humain deux réponses générées par le modèle (A et B). L'humain doit désigner laquelle est la meilleure (**Accepted**) et laquelle est la moins bonne (**Rejected**). 

> [!NOTE]
🔑 **Je dois insister sur la richesse de cette donnée :** Ce n'est pas seulement binaire. Souvent, les annotateurs expliquent *pourquoi* ils préfèrent A à B (ex: "A est plus poli", "B contient une erreur de fait"). Cette base de données de comparaisons est le trésor de guerre des grandes entreprises d'IA.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On ne cherche pas des réponses "vraies" ou "fausses". On cherche des nuances de qualité. 

> Parfois, les deux réponses sont bonnes, mais l'une est plus concise. Parfois, les deux sont mauvaises, mais l'une est moins toxique. Le RLHF apprend au modèle à naviguer dans ce "gris" sémantique.

---
## Acte II : L'entraînement du Reward Model (Le Juge)
Une fois que nous avons nos milliers de paires de préférences humaines, nous allons créer un modèle capable de prédire ce que l'humain aurait choisi. C'est le **Reward Model (RM)**.

Regardons la **Figure 12-7 : Transformer un LLM en Reward Model** .

{{< bookfig src="281.png" week="12" >}}

**Explication de la Figure 12-7** : Pour créer ce juge, on prend une copie de notre modèle SFT et on effectue une modification chirurgicale. On retire la "tête" de prédiction de mots (la LM Head qui choisit le prochain token parmi 50 000 choix) et on la remplace par une **tête de classification de qualité** (une couche linéaire qui ne sort qu'un seul chiffre : un score scalaire). 

❓ **Comment le Reward Model apprend-il ?**
On lui présente les paires (A, B) de notre dataset de préférences. Sa mission est simple : le score qu'il attribue à la réponse "Acceptée" doit être supérieur au score de la réponse "Rejetée". 

🔢 **La mathématique du RM :** On utilise une fonction de perte contrastive. Le modèle est puni s'il donne un meilleur score à la mauvaise réponse. À la fin de cette phase, nous avons un "Juge automatique" qui peut lire n'importe quel texte et dire : "C'est une réponse de qualité 0.85" ou "C'est une réponse de qualité 0.12". 

Regardez la **Figure 12-8 : Utilisation du Reward Model** . Elle montre le RM en action : il prend un prompt et une génération, et il produit une note. Ce chiffre unique devient la "récompense" qui va guider l'étape suivante.

### Laboratoire de code : Structure d'un Reward Model simple
Voici comment nous pourrions initialiser un tel modèle en utilisant BERT comme base pour le jugement :

```python
# Testé sur Colab T4 16GB VRAM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. On utilise un modèle de représentation (BERT) car il est excellent pour juger
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. On configure num_labels=1 pour avoir une sortie scalaire unique (le score)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1 
).to("cuda")

# 3. Exemple de scoring
text = "Assistant response: I can help you with your question..."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    score = reward_model(**inputs).logits
    print(f"Score de récompense : {score.item():.4f}")
```

---
## Acte III : L'optimisation par renforcement (PPO)
Nous arrivons au sommet de la montagne. Nous avons notre modèle de base (le SFT) et notre juge (le RM). Nous allons maintenant faire "jouer" le modèle pour qu'il s'améliore.

Pour cela, nous utilisons un algorithme d'apprentissage par renforcement profond (*Deep Reinforcement Learning*) appelé **PPO** (*Proximal Policy Optimization*). Regardez la **Figure 12-9 : Le cycle de l'apprentissage par renforcement** .

{{< bookfig src="286.png" week="12" >}}

**Explication de la Figure 12-9** : C'est une boucle itérative. 
1.  Le modèle (appelé ici la **Policy**) génère une réponse à un prompt.
2.  Le Reward Model lit la réponse et donne une note (la récompense).
3.  L'algorithme PPO ajuste les poids de la Policy pour que, la prochaine fois, elle génère une réponse qui obtiendra une meilleure note.

> [!IMPORTANT]
🔑 **Je dois insister sur la difficulté de cette étape :** Le PPO est notoirement instable. 

> Si vous laissez le modèle courir après les scores sans garde-fou, il va découvrir des "failles" dans le Reward Model. Il pourrait commencer à répondre par des suites de mots absurdes qui "plaisent" mathématiquement au juge mais n'ont aucun sens pour un humain. C'est ce qu'on appelle le **Reward Hacking**.

---
## La sécurité avant tout : La divergence KL et les modèles multiples
Pour empêcher le modèle de devenir fou pendant le PPO, nous ajoutons une "laisse" de sécurité. 

1.  **La divergence KL (Kullback-Leibler)** : On garde une copie gelée du modèle SFT original (le modèle "Référence"). Pendant que le modèle PPO apprend, on calcule à quel point ses probabilités s'éloignent du modèle de référence. S'il s'éloigne trop (s'il commence à parler de manière bizarre), on lui inflige une pénalité. 

> [!NOTE]
💡 **L'intuition :** Le modèle doit devenir "meilleur", mais il doit rester un modèle de langage cohérent.

2.  **Modèles de récompense multiples** : Comme le montre la **Figure 12-10**, on ne se contente pas d'un seul juge.
    *   Un juge note l'**Utilité** (*Helpfulness*).
    *   Un autre juge (souvent plus sévère) note la **Sécurité** (*Safety*).

{{< bookfig src="287.png" week="12" >}}

La récompense finale est une combinaison des deux. C'est ainsi que l'on évite qu'une réponse soit "très utile mais extrêmement dangereuse".

---
## Éthique et Responsabilité : Les pièges du RLHF

> [!CAUTION]
⚖️ Mes chers étudiants, le RLHF est une technologie puissante, mais elle est fragile et potentiellement biaisée.

1.  **La Loi de Goodhart et l'optimisation excessive** : À force de vouloir maximiser un score de "politesse", le modèle finit par devenir mielleux et hypocrite. Il s'excuse sans cesse au lieu de répondre. ❌ **C'est un échec d'alignement.**
2.  **Le coût humain de l'étiquetage** : Les milliers d'humains qui trient les réponses de "Sécurité" sont souvent exposés à des contenus traumatisants (violence, haine) pour apprendre au modèle à les rejeter. La responsabilité de l'ingénieur inclut aussi la protection de ces travailleurs de la donnée.
3.  **L'uniformisation de la pensée** : En alignant le modèle sur un "humain moyen", on risque d'effacer les nuances culturelles. Qui décide de ce qui est une "bonne" réponse à une question philosophique ou politique ?

> [!TIP]
✉️ **Mon message** : Le RLHF est ce qui a donné aux LLMs leur vernis de civilisation.

> Mais n'oubliez jamais que derrière l'algorithme PPO, il y a des choix humains subjectifs. Le RLHF ne crée pas de vérité, il crée un consensus statistique sur la désirabilité sociale d'un discours.

---
Vous maîtrisez maintenant le concept monumental du RLHF. Vous comprenez comment un juge artificiel peut guider un écrivain numérique. Cependant, le RLHF est lourd, instable et coûteux. Dans la prochaine section ➡️, nous allons découvrir une alternative révolutionnaire qui simplifie tout cela radicalement : le **DPO**. Vous allez voir, c'est d'une élégance mathématique rare.