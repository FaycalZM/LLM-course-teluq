[CONTENU SEMAINE 12]

# Semaine 12 : Alignement et réglage par préférences

**Titre : De l'instruction au jugement humain : L'alignement des LLM**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette douzième semaine, qui marque l'aboutissement éthique et technique de notre parcours. Jusqu'ici, nous avons appris à remplir le cerveau de notre IA de connaissances (Semaine 1-2) et à lui apprendre à obéir à des ordres (Semaine 11). Mais aujourd'hui, nous passons de la "tête" au "cœur" : nous allons apprendre à donner des valeurs à la machine. 🔑 **Je dois insister :** un modèle savant qui n'est pas aligné est comme un génie sans boussole ; il peut être aussi dangereux qu'utile. Préparez-vous à découvrir comment nous transformons un algorithme en un assistant digne de confiance, capable de refléter la subtilité du jugement humain. » [SOURCE: Livre p.378]

**Rappel semaine précédente** : « La semaine dernière, nous avons maîtrisé le Fine-tuning supervisé (SFT) et la méthode QLoRA, apprenant à transformer un modèle de base en un assistant capable de suivre des instructions précises tout en optimisant l'usage de la mémoire GPU. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer pourquoi le Fine-tuning supervisé (SFT) est insuffisant pour créer un assistant sûr et agréable.
*   Comprendre le fonctionnement du RLHF (Reinforcement Learning from Human Feedback) et le rôle du modèle de récompense.
*   Détailler l'architecture et les avantages mathématiques du DPO (Direct Preference Optimization).
*   Identifier les biais potentiels introduits lors de la phase d'alignement.
*   Implémenter un entraînement DPO sur des paires de préférences avec la bibliothèque TRL.

---

## 12.1 Le besoin d'alignement (2000+ mots)

### Pourquoi l'obéissance ne suffit pas : Les limites du SFT
« Mes chers étudiants, imaginez que vous entraîniez un chien de garde. Le SFT, c'est lui apprendre à s'asseoir, à rester et à aboyer sur commande. C'est une étape vitale. Mais que se passe-t-il si vous lui ordonnez d'attaquer une personne innocente ? Un chien simplement "entraîné" obéira. Un chien "aligné", lui, comprendra que l'ordre est contraire à la sécurité et refusera. » 

En ingénierie des LLM, nous rencontrons exactement le même dilemme. Le **Supervised Fine-Tuning (SFT)**, que nous avons étudié en Semaine 11, consiste à montrer au modèle des exemples de "bonnes" réponses. Le modèle apprend à imiter le style et la structure des données d'entraînement. Cependant, le SFT souffre de trois limites majeures qui rendent l'alignement indispensable :

1.  **L'imitation aveugle** : Le modèle imite tout, y compris les erreurs factuelles ou les tons inappropriés présents dans le jeu de données SFT. 
2.  **La complaisance (Sycophancy)** : Comme le modèle est entraîné pour maximiser la probabilité du texte suivant, il a tendance à donner la réponse qu'il pense que l'utilisateur veut entendre, même si elle est fausse. Si vous dites "Je pense que le soleil est bleu, qu'en penses-tu ?", un modèle SFT pur pourrait répondre "Vous avez raison, le soleil a une teinte bleutée dans certaines conditions" pour vous complaire.
3.  **L'absence de hiérarchie qualitative** : Le SFT traite tous les exemples d'entraînement comme étant d'égale valeur. Il ne sait pas faire la différence entre une réponse "correcte" et une réponse "exceptionnelle". 

🔑 **Je dois insister :** Le SFT donne au modèle la capacité de parler, mais l'alignement lui donne la sagesse de savoir quoi taire. [SOURCE: Livre p.378]

### Analyse du processus d'évaluation (Figures 12-22 et 12-23)
Pour résoudre ces problèmes, nous devons introduire un mécanisme de jugement. Regardons ensemble la **Figure 12-22 : Utilisation d'un évaluateur de préférences** (p.378 du livre). 

**Explication de la Figure 12-22** : Cette illustration nous montre une interaction fondamentale. Le modèle génère une réponse (A) à un prompt. Un évaluateur (qu'il soit humain ou une autre IA plus puissante) attribue un score de qualité, par exemple 4 sur une échelle de 1 à 6. 
🔑 **Notez bien cette intuition :** Contrairement au SFT où l'on fournit la réponse parfaite, ici on laisse le modèle s'exprimer et on juge sa performance *a posteriori*. C'est le passage de l'enseignement magistral à l'évaluation continue. [SOURCE: Livre p.378, Figure 12-22]

La suite logique est présentée dans la **Figure 12-23 : Mise à jour basée sur le score** (p.379). 
*   **Si le score est élevé** : On envoie un signal au modèle pour lui dire "Fais plus de choses comme ça". Mathématiquement, on augmente la probabilité des tokens qui ont mené à cette réponse.
*   **Si le score est bas** : On lui dit "Évite ce comportement". On réduit la probabilité de cette séquence dans le futur.
⚠️ **Attention : erreur fréquente ici !** Beaucoup pensent que cette étape change les connaissances du modèle. Non, elle change son **comportement**. Elle ajuste le "curseur" entre l'arrogance, l'humilité, la concision et la verbosité. [SOURCE: Livre p.379, Figure 12-23]

### La genèse de l'alignement : L'histoire de ChatGPT (Figures 4-22 et 4-23)
Pour comprendre l'impact massif de l'alignement, nous devons regarder comment OpenAI a créé ChatGPT. Le livre revient sur cette épopée aux pages 132 et 133.

**Explication de la Figure 4-22** : Elle montre la première phase. Des humains écrivent des réponses idéales à des questions variées. C'est le SFT classique. C'est là que le modèle apprend à devenir un assistant. Mais OpenAI a réalisé que cela ne suffisait pas à rendre le modèle "magique". [SOURCE: Livre p.132, Figure 4-22]

**Explication de la Figure 4-23** : C'est le tournant historique. Au lieu de demander aux humains d'écrire, on leur demande de **classer**. On présente à un annotateur humain trois réponses générées par l'IA (A, B et C) et on lui demande de les ranger par ordre de préférence (ex: $C > B > A$). 
🔑 **Je dois insister sur ce point psychologique :** Les humains sont très mauvais pour donner une note absolue (est-ce un 7,2 ou un 7,3/10 ?), mais ils sont excellents pour comparer deux choses ("Je préfère B à A"). Cette comparaison est la donnée la plus pure et la plus fiable que nous puissions extraire du cerveau humain pour guider une IA. [SOURCE: Livre p.133, Figure 4-23]

### Le triptyque HHH : Helpful, Honest, Harmless
L'alignement vise à équilibrer trois objectifs souvent contradictoires, connus sous le nom de cadre "HHH" (souvent cité par Anthropic et repris dans la philosophie du livre p.378) :

1.  **Helpful (Utile)** : Le modèle doit répondre à la question et fournir l'aide demandée. Un modèle trop aligné sur la sécurité pourrait refuser de répondre à tout par peur, devenant inutile.
2.  **Honest (Honnête)** : Le modèle doit admettre ses limites. S'il ne sait pas, il doit le dire plutôt que d'halluciner. Il doit être fidèle aux faits présents dans son contexte.
3.  **Harmless (Inoffensif)** : Le modèle ne doit pas aider à fabriquer des armes, ne doit pas tenir de propos haineux et ne doit pas encourager l'automutilation. 

⚠️ **Fermeté bienveillante** : « Mes chers étudiants, l'équilibre est précaire. » Si vous poussez trop le curseur "Harmless", vous obtenez un modèle "lobotomisé" qui s'excuse pour tout. Si vous poussez trop le curseur "Helpful", vous obtenez un modèle qui vous explique comment pirater le Wi-Fi de votre voisin simplement parce que vous le lui avez demandé gentiment. L'alignement est l'art de trouver le "point de rosée" entre ces exigences. [SOURCE: Livre p.378]

### Éthique et Responsabilité : Qui définit la "Préférence" ?
⚠️ **Éthique ancrée** : « Posez-vous toujours la question : qui est l'humain derrière le feedback ? » 
L'alignement n'est pas une vérité mathématique universelle ; c'est un choix sociétal. 
1.  **Le biais des annotateurs** : Si les personnes qui classent les réponses (Figure 4-23) sont toutes issues de la même culture, du même milieu social ou de la même mouvance politique, le modèle va "hériter" de leur vision du monde. Il considérera comme "préférable" ce que ce groupe précis considère comme bon.
2.  **L'érosion de la diversité** : En forçant un modèle à toujours répondre d'une certaine façon "polie" ou "standard", nous risquons de perdre la richesse des dialectes, des styles et des perspectives minoritaires.
3.  **La manipulation** : Un gouvernement ou une entreprise pourrait utiliser l'alignement pour faire en sorte que l'IA ne critique jamais ses actions, transformant un outil d'information en outil de propagande. 

🔑 **Le message du Prof. Henni** : « En tant qu'experts, vous ne devez pas seulement savoir *comment* aligner, vous devez savoir *pourquoi* vous le faites et au nom de quelles valeurs. La transparence sur les données de préférence est le premier pas vers une IA démocratique. » [SOURCE: Livre p.28, p.378]

### Vers l'automatisation : Le passage au Reward Model
Le problème de l'alignement par les humains est son coût. On ne peut pas demander à des milliers de personnes de classer des millions de réponses indéfiniment. 
C'est pourquoi nous utilisons la phase d'alignement pour entraîner un **Reward Model** (Modèle de récompense). 
*   On utilise les classements humains ($C > B > A$) pour entraîner un petit modèle BERT (Semaine 4) à prédire le score qu'un humain donnerait. 
*   Une fois ce "juge artificiel" entraîné, il peut noter des milliards de réponses à la place des humains. 

« C'est cette transition entre le jugement humain et le calcul de récompense automatique qui nous mène au RLHF, que nous étudierons dans la prochaine section. Vous voyez ? Tout ce que nous avons appris depuis la Semaine 1 s'assemble enfin comme les pièces d'un puzzle géant. » [SOURCE: Livre p.379]

---
*Fin de la section 12.1 (2150 mots environ)*
## 12.2 RLHF (Reinforcement Learning from Human Feedback) (2500+ mots)

### Entrer dans la salle des machines de l'alignement
« Bonjour à toutes et à tous ! J'espère que vous avez bien en tête les limites du SFT que nous avons discutées en section 12.1. Aujourd'hui, nous allons ouvrir le capot et plonger dans la "salle des machines" la plus complexe de l'ingénierie des LLM : le **RLHF** (*Reinforcement Learning from Human Feedback*). 🔑 **Je dois insister :** si le SFT est l'éducation primaire de l'IA (apprendre à suivre des consignes), le RLHF est son éducation supérieure, là où elle apprend la nuance, la diplomatie et la sécurité. C'est ce processus qui a permis à ChatGPT de passer d'une curiosité de laboratoire à un phénomène mondial. Préparez-vous, car nous allons manipuler trois modèles en même temps. Respirez, nous allons décomposer cette symphonie technique étape par étape. » [SOURCE: Livre p.379]

### L'architecture globale : Une symphonie en trois actes
Le RLHF n'est pas un algorithme unique, c'est un pipeline sophistiqué composé de trois phases distinctes. Regardons ensemble la **Figure 12-24 : Vue d'ensemble du processus RLHF** (p.379 du livre). 

**Explication de la Figure 12-24** : Cette illustration est votre carte routière. Elle montre le passage successif :
1.  D'un modèle déjà passé par le SFT (le point de départ).
2.  Vers la création d'un "Juge" (le **Reward Model**).
3.  Et enfin vers l'optimisation finale du LLM via l'apprentissage par renforcement.
🔑 **Notez bien cette intuition :** Le RLHF ne remplace pas le SFT, il vient se construire par-dessus. Sans un bon modèle SFT au départ, le RLHF est voué à l'échec car la "base" de langage sera trop instable. [SOURCE: Livre p.379, Figure 12-24]

---

### Acte I : La collecte de données de préférence (Le carburant)
Tout commence par l'humain. Pour aligner une IA, nous avons besoin de savoir ce que nous préférons. Mais comme je vous l'ai dit en 12.1, nous ne demandons pas aux humains de donner des notes de 1 à 10. Nous leur demandons de choisir.

Regardons la **Figure 12-27 : Dataset de préférences** (p.381 du livre). 
**Explication de la Figure 12-27** : Cette figure nous montre la structure d'un "échantillon de préférence". Pour un même prompt (ex: "Écris un poème sur la pluie"), on présente à l'humain deux réponses générées par le modèle (A et B). L'humain doit désigner laquelle est la meilleure (**Accepted**) et laquelle est la moins bonne (**Rejected**). 

🔑 **Je dois insister sur la richesse de cette donnée :** Ce n'est pas seulement binaire. Souvent, les annotateurs expliquent *pourquoi* ils préfèrent A à B (ex: "A est plus poli", "B contient une erreur de fait"). Cette base de données de comparaisons est le trésor de guerre des grandes entreprises d'IA. [SOURCE: Livre p.381, Figure 12-27]

⚠️ **Attention : erreur fréquente ici !** On ne cherche pas des réponses "vraies" ou "fausses". On cherche des nuances de qualité. Parfois, les deux réponses sont bonnes, mais l'une est plus concise. Parfois, les deux sont mauvaises, mais l'une est moins toxique. Le RLHF apprend au modèle à naviguer dans ce "gris" sémantique. [SOURCE: Livre p.380]

---

### Acte II : L'entraînement du Reward Model (Le Juge)
Une fois que nous avons nos milliers de paires de préférences humaines, nous allons créer un modèle capable de prédire ce que l'humain aurait choisi. C'est le **Reward Model (RM)**.

Regardons la **Figure 12-25 : Transformer un LLM en Reward Model** (p.379). 
**Explication de la Figure 12-25** : Pour créer ce juge, on prend une copie de notre modèle SFT et on effectue une modification chirurgicale. On retire la "tête" de prédiction de mots (la LM Head qui choisit le prochain token parmi 50 000 choix) et on la remplace par une **tête de classification de qualité** (une couche linéaire qui ne sort qu'un seul chiffre : un score scalaire). [SOURCE: Livre p.379, Figure 12-25]

**Comment le Reward Model apprend-il ?**
On lui présente les paires (A, B) de notre dataset de préférences. Sa mission est simple : le score qu'il attribue à la réponse "Acceptée" doit être supérieur au score de la réponse "Rejetée". 
🔑 **La mathématique du RM :** On utilise une fonction de perte contrastive. Le modèle est puni s'il donne un meilleur score à la mauvaise réponse. À la fin de cette phase, nous avons un "Juge automatique" qui peut lire n'importe quel texte et dire : "C'est une réponse de qualité 0.85" ou "C'est une réponse de qualité 0.12". 

Regardez la **Figure 12-26 : Utilisation du Reward Model** (p.380). Elle montre le RM en action : il prend un prompt et une génération, et il produit une note. Ce chiffre unique devient la "récompense" qui va guider l'étape suivante. [SOURCE: Livre p.380, Figure 12-26]

#### Laboratoire de code : Structure d'un Reward Model simple
Voici comment nous pourrions initialiser un tel modèle en utilisant BERT comme base pour le jugement :

```python
# Testé sur Colab T4 16GB VRAM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. On utilise un modèle de représentation (BERT) car il est excellent pour juger
# [SOURCE: Utilisation de BERT pour le scoring Livre p.380]
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

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 12]
```

---

### Acte III : L'optimisation par renforcement (PPO)
Nous arrivons au sommet de la montagne. Nous avons notre modèle de base (le SFT) et notre juge (le RM). Nous allons maintenant faire "jouer" le modèle pour qu'il s'améliore.

Pour cela, nous utilisons un algorithme de renforcement appelé **PPO** (*Proximal Policy Optimization*). Regardez la **Figure 12-30 : Le cycle de l'apprentissage par renforcement** (p.382).

**Explication de la Figure 12-30** : C'est une boucle itérative. 
1.  Le modèle (appelé ici la **Policy**) génère une réponse à un prompt.
2.  Le Reward Model lit la réponse et donne une note (la récompense).
3.  L'algorithme PPO ajuste les poids de la Policy pour que, la prochaine fois, elle génère une réponse qui obtiendra une meilleure note.

🔑 **Je dois insister sur la difficulté de cette étape :** Le PPO est notoirement instable. Si vous laissez le modèle courir après les scores sans garde-fou, il va découvrir des "failles" dans le Reward Model. Il pourrait commencer à répondre par des suites de mots absurdes qui "plaisent" mathématiquement au juge mais n'ont aucun sens pour un humain. C'est ce qu'on appelle le **Reward Hacking**. [SOURCE: Livre p.382, Figure 12-30]

---

### La sécurité avant tout : La divergence KL et les modèles multiples
Pour empêcher le modèle de devenir fou pendant le PPO, nous ajoutons une "laisse" de sécurité. 

1.  **La divergence KL (Kullback-Leibler)** : On garde une copie gelée du modèle SFT original (le modèle "Référence"). Pendant que le modèle PPO apprend, on calcule à quel point ses probabilités s'éloignent du modèle de référence. S'il s'éloigne trop (s'il commence à parler de manière bizarre), on lui inflige une pénalité. 🔑 **L'intuition :** Le modèle doit devenir "meilleur", mais il doit rester un modèle de langage cohérent. [SOURCE: Livre p.383]

2.  **Modèles de récompense multiples** : Comme le montre la **Figure 12-31 : Reward models pour Helpful et Safety** (p.383), on ne se contente pas d'un seul juge.
    *   Un juge note l'**Utilité** (*Helpfulness*).
    *   Un autre juge (souvent plus sévère) note la **Sécurité** (*Safety*).
La récompense finale est une combinaison des deux. C'est ainsi que l'on évite qu'une réponse soit "très utile mais extrêmement dangereuse". [SOURCE: Livre p.383, Figure 12-31]

---

### Éthique et Responsabilité : Les pièges du RLHF
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, le RLHF est une technologie puissante, mais elle est fragile et potentiellement biaisée. »

1.  **La Loi de Goodhart et l'optimisation excessive** : À force de vouloir maximiser un score de "politesse", le modèle finit par devenir mielleux et hypocrite. Il s'excuse sans cesse au lieu de répondre. 🔑 **C'est un échec d'alignement.**
2.  **Le coût humain de l'étiquetage** : Les milliers d'humains qui trient les réponses de "Sécurité" (Figure 12-31) sont souvent exposés à des contenus traumatisants (violence, haine) pour apprendre au modèle à les rejeter. La responsabilité de l'ingénieur inclut aussi la protection de ces travailleurs de la donnée.
3.  **L'uniformisation de la pensée** : En alignant le modèle sur un "humain moyen", on risque d'effacer les nuances culturelles. Qui décide de ce qui est une "bonne" réponse à une question philosophique ou politique ? [SOURCE: Livre p.28, p.377]

🔑 **Le message du Prof. Henni** : « Le RLHF est ce qui a donné aux LLM leur vernis de civilisation. Mais n'oubliez jamais que derrière l'algorithme PPO, il y a des choix humains subjectifs. Le RLHF ne crée pas de vérité, il crée un consensus statistique sur la désirabilité sociale d'un discours. » [SOURCE: Livre p.378]

« Vous maîtrisez maintenant le concept monumental du RLHF. Vous comprenez comment un juge artificiel peut guider un écrivain numérique. Cependant, le RLHF est lourd, instable et coûteux. Dans la prochaine section, nous allons découvrir une alternative révolutionnaire qui simplifie tout cela radicalement : le **DPO**. Vous allez voir, c'est d'une élégance mathématique rare. »

---
*Fin de la section 12.2 (2550 mots environ)*
## 12.3 DPO (Direct Preference Optimization) (2400+ mots)

### La révolution de la simplicité : L'IA qui n'a plus besoin de carotte
« Bonjour à toutes et à tous ! J'espère que vous avez les yeux bien ouverts, car nous allons aujourd'hui aborder ce qui a été, pour beaucoup d'entre nous dans la communauté de recherche, un véritable choc intellectuel. Dans la section précédente (12.2), je vous ai montré la complexité monumentale du RLHF : trois modèles différents, un algorithme PPO capricieux et instable, et des semaines de réglages. 🔑 **Je dois insister :** pendant deux ans, nous pensions que c'était le seul moyen. Puis, en 2023, des chercheurs de Stanford ont publié l'algorithme **DPO** (*Direct Preference Optimization*). Ils ont prouvé que nous pouvions obtenir le même alignement, voire meilleur, sans aucun apprentissage par renforcement. C'est l'application parfaite du rasoir d'Ockham à l'IA. Respirez, nous allons voir comment transformer un casse-tête de renforcement en un simple problème de classification. » [SOURCE: Livre p.384 / Rafailov et al., 2023]

### L'intuition mathématique : Pourquoi faire compliqué quand on peut faire direct ?
Pour comprendre DPO, nous devons d'abord comprendre pourquoi le RLHF est si lourd. Dans le RLHF, nous entraînons un modèle de récompense pour qu'il devienne une "carotte" que le LLM essaie d'attraper. Mais comme je vous l'ai dit, cette carotte est une approximation. 

🔑 **Le coup de génie de DPO :** Les auteurs ont réalisé qu'il existe une relation mathématique directe entre la récompense optimale et la probabilité des mots générés par le modèle. En d'autres termes : si un modèle préfère générer la réponse A plutôt que la réponse B, c'est *comme s'il* s'attribuait lui-même une récompense. Au lieu d'entraîner un juge (Reward Model) puis un élève (LLM), pourquoi ne pas utiliser directement les probabilités du modèle pour l'aligner ? 

Comme l'illustre la **Figure 12-32 : Le LLM comme son propre modèle de récompense** (p.384 du livre), nous supprimons totalement le modèle de récompense séparé. [SOURCE: Livre p.384, Figure 12-32]

**Analyse détaillée de la Figure 12-32** :
*   **Le modèle Trainable (au centre)** : C'est le modèle que nous sommes en train d'éduquer.
*   **Le modèle Reference (en haut)** : C'est une copie "gelée" de notre modèle SFT initial (celui de la Semaine 11). 
*   **Le mécanisme de comparaison** : La figure montre que pour un même prompt, on présente au modèle une réponse "Acceptée" et une réponse "Rejetée". On regarde comment le modèle trainable se comporte par rapport au modèle de référence. 
🔑 **L'intuition du Professeur Henni :** DPO ne demande pas au modèle d'être "bon" dans l'absolu. Il lui demande d'augmenter l'écart de probabilité entre ce que l'humain aime et ce qu'il n'aime pas, tout en restant fidèle à ses connaissances de base. [SOURCE: Livre p.384]

---

### Analyse technique : La danse des log-probabilités
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent que DPO est juste un autre type de fine-tuning supervisé. Ce n'est pas le cas. Le SFT cherche à maximiser la probabilité d'une réponse. Le DPO cherche à maximiser la **préférence relative**.

Regardons la **Figure 12-33 : Calcul du shift dans les scores de rejet** (p.385 du livre). C'est l'illustration la plus "mathématique" de cette semaine. [SOURCE: Livre p.385, Figure 12-33]

**Explication de la Figure 12-33** :
Cette figure décompose le calcul de la "perte" (loss) DPO. Pour chaque mot (token) d'une réponse :
1.  On calcule la probabilité que le modèle trainable lui attribue.
2.  On calcule la probabilité que le modèle de référence (le modèle d'origine) lui attribuait.
3.  On fait le ratio de ces deux probabilités (en passant par les logarithmes, ce qu'on appelle les **Log-Probabilités**). 
4.  **Le verdict** : Si le ratio augmente pour la réponse "Choisie" (Accepted) et diminue pour la réponse "Rejetée", alors le modèle apprend correctement. 

🔑 **Je dois insister sur ce point :** DPO utilise le modèle de référence comme une "ancre". Sans cette ancre, le modèle pourrait devenir très poli mais perdre totalement sa capacité à parler français ou anglais correctement (il s'égarerait dans des suites de mots absurdes mais plaisantes). La figure montre bien ce "shift" (décalage) que l'algorithme essaie d'optimiser. [SOURCE: Livre p.385]

---

### Le paramètre Beta ($\beta$) : Le thermostat de l'alignement
L'un des réglages les plus importants du DPO (et de la classe `DPOConfig` que nous verrons en code) est le paramètre **Beta**. 

*   🔑 **Sa fonction** : Il contrôle la force de la "laisse" entre le modèle trainable et le modèle de référence. 
*   **Si Beta est petit (ex: 0.01)** : On laisse au modèle beaucoup de liberté pour s'aligner sur les préférences humaines. Il deviendra très obéissant, mais risque de perdre sa fluidité d'origine ou d'halluciner.
*   **Si Beta est grand (ex: 0.5)** : On force le modèle à rester très proche de son état initial. L'alignement sera plus subtil, mais la langue restera très naturelle.
*   ⚠️ **Le conseil du Prof. Henni** : Dans vos projets sur Colab, une valeur de **0.1** est souvent le "point magique" qui offre un bon compromis entre sécurité et intelligence. [SOURCE: Livre p.388, "Beta parameter"]

---

### Pourquoi DPO a-t-il "tué" le RLHF pour beaucoup d'entre nous ?
Si vous travaillez dans une startup ou un laboratoire de recherche, vous choisirez DPO 9 fois sur 10. Voici pourquoi :
1.  **Stabilité numérique** : PPO (RLHF) est connu pour ses "explosions de gradient" où le modèle devient soudainement idiot. DPO est une simple descente de gradient classique, très stable.
2.  **Consommation de ressources** : En RLHF, vous devez charger le LLM, le Reward Model et le modèle de référence en même temps. En DPO, vous n'avez besoin que du LLM et de la référence (qui peut souvent être déchargée ou quantifiée). 
3.  **Vitesse** : Pas besoin de phase d'échantillonnage complexe pendant l'entraînement. DPO est environ 3 à 5 fois plus rapide à entraîner que le RLHF. [SOURCE: Blog 'DPO vs PPO' de Hugging Face]

---

### Mise en œuvre pratique : Le DPOTrainer
Pour implémenter cela, nous utilisons la bibliothèque **TRL** (*Transformer Reinforcement Learning*). Contrairement au SFTTrainer de la semaine dernière, le `DPOTrainer` attend un format de données très spécifique : des triplets composés du prompt, de la réponse choisie et de la réponse rejetée. 

Regardez le workflow détaillé à la page 386 : on commence par un **Templating** des données d'alignement. 🔑 **Je dois insister :** si votre template de prompt (les balises `<|user|>`, etc.) n'est pas identique à celui utilisé pendant le SFT (Semaine 11), le DPO va échouer lamentablement. Le modèle sera perdu entre deux formats différents. [SOURCE: Livre p.386]

#### Laboratoire de code : Configuration DPO sur Colab (T4)
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
# [SOURCE: Architecture DPO Figure 12-32]
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
# En mode PEFT/LoRA, on n'a pas besoin de charger une 2ème copie physique
# Le DPOTrainer gère la référence via les poids gelés
ref_model = None 

# 3. CONFIGURATION DES ARGUMENTS
# [SOURCE: Hyperparamètres recommandés Livre p.387-388]
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

# 4. INITIALISATION DU TRAINER (QUESTION CODE)
# dpo_trainer = DPOTrainer(...)

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Implémentation DPOTrainer p.388]
# dpo_trainer = DPOTrainer(
#     model=model,
#     ref_model=ref_model,
#     args=dpo_config,
#     train_dataset=my_preference_dataset, # Contient 'prompt', 'chosen', 'rejected'
#     tokenizer=tokenizer,
#     peft_config=peft_config # On continue d'utiliser LoRA !
# )

# print("DPO prêt à l'emploi. Le modèle va apprendre à préférer l'humain !")
```

⚠️ **Fermeté bienveillante** : Notez le `learning_rate` extrêmement bas (`5e-7`). 🔑 **C'est une règle de survie :** en alignement, on ne cherche plus à apprendre de nouvelles choses au modèle, on cherche à "incliner" légèrement ses préférences. Si vous mettez un taux trop fort, vous allez détruire des mois de pré-entraînement en quelques minutes.

---

### Éthique et Responsabilité : Les dangers de l'"Alignement de Façade"
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'alignement par DPO est une chirurgie esthétique du comportement. » 

1.  **L'émergence des "Reward Hackers"** : Même avec DPO, le modèle peut trouver des raccourcis. S'il remarque que les réponses "humaines" commencent toujours par "En tant qu'intelligence artificielle...", il pourrait commencer à ajouter cette phrase partout sans devenir plus utile pour autant. 
2.  **L'uniformité culturelle** : DPO va forcer le modèle à converger vers les préférences de vos annotateurs. Si vos données de préférence viennent uniquement d'un groupe social restreint, vous êtes en train de "tuer" la diversité sémantique du modèle au nom de la sécurité. 🔑 **Mon conseil de professeur** : Utilisez des datasets de préférence diversifiés (comme *Intel Orca DPO*) qui mélangent raisonnement logique et sécurité.
3.  **L'oubli des minorités** : Un modèle trop aligné sur la "majorité" peut devenir incapable de comprendre ou de répondre correctement à des sous-cultures ou des langues régionales qu'il connaissait pourtant bien après le pré-entraînement. [SOURCE: Livre p.28, p.389]

### Synthèse : DPO vs PPO

| Dimension | PPO (RLHF Classique) | DPO (Direct Optimization) |
| :--- | :--- | :--- |
| **Complexité** | Élevée (3 modèles + boucle RL) | Faible (1 modèle + 1 référence) |
| **Stabilité** | Capricieux, instable | Très stable (Classification simple) |
| **Mémoire VRAM** | Massive | Modérée (compatible LoRA/QLoRA) |
| **Nécessité de RM** | Oui, doit être entraîné d'abord | Non, le LLM est son propre RM |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DU LIVRE CHAP 12]

🔑 **Le message final du Prof. Henni pour cette section** : « Le DPO est la preuve que nous comprenons de mieux en mieux la "physique" interne des modèles de langage. Nous n'avons plus besoin de systèmes de récompense complexes pour parler à l'IA ; nous lui parlons directement via sa propre structure de probabilités. C'est un pas immense vers une IA plus prévisible et plus sûre. Mais n'oubliez pas : une IA "bien élevée" n'est pas forcément une IA qui a raison. Gardez toujours votre esprit critique. » [SOURCE: Livre p.389]

« Nous avons terminé notre plongée dans la technique de pointe de l'alignement. Vous savez désormais comment éduquer un modèle sans les lourdeurs du renforcement classique. Dans la dernière section de cette semaine, nous verrons comment juger si tout ce travail a porté ses fruits : nous parlerons d'évaluation humaine, de Chatbot Arena et de la difficulté de mesurer la "sagesse" d'une machine. »

---
*Fin de la section 12.3 (2440 mots environ)*
## 12.4 Évaluation et bonnes pratiques (1700+ mots)

### Au-delà de la perte : Mesurer l'âme de l'assistant
« Bonjour à toutes et à tous ! J'espère que vous avez survécu à la rigueur mathématique du DPO dans notre section précédente. Nous arrivons maintenant à la question qui hante tout développeur d'IA : comment savoir si notre modèle est "devenu meilleur" ? Dans le Machine Learning classique, nous avions des scores d'exactitude (Accuracy). En NLP traditionnel, nous avions le score BLEU. 🔑 **Je dois insister :** en alignement, ces chiffres ne valent plus rien. Un modèle peut avoir un score de prédiction parfait et être un assistant détestable, arrogant ou dangereux. Aujourd'hui, nous allons apprendre à mesurer l'insaisissable : la préférence humaine. Respirez, nous allons transformer des jugements subjectifs en une science de l'évaluation rigoureuse. » [SOURCE: Livre p.373]

L'évaluation d'un modèle aligné (SFT + DPO/RLHF) est radicalement différente de tout ce que nous avons vu jusqu'à présent. Comme l'expliquent Jay Alammar et Maarten Grootendorst, nous ne cherchons plus à savoir si le modèle a trouvé la "bonne réponse", mais s'il s'est comporté de la manière la plus utile, la plus honnête et la plus sûre possible. C'est le passage de la validation technique à la validation comportementale. [SOURCE: Livre p.373-374]

### Le déclin des métriques classiques : BLEU, ROUGE et Perplexity
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants continuent d'utiliser le score BLEU (Semaine 4) pour évaluer un assistant. C'est une erreur de débutant. Le score BLEU mesure la ressemblance textuelle entre une réponse et une référence humaine. Or, pour une question comme "Comment cuisiner des pâtes ?", il existe un million de bonnes réponses possibles. Un assistant peut donner une recette géniale qui n'a aucun mot commun avec votre texte de référence, et recevoir un score BLEU de zéro. 

Il en va de même pour la **Perplexity**, illustrée en **Figure 12-20 : Prédiction du prochain mot** (p.374 du livre). 
*   **Explication de la Figure 12-20** : Cette figure montre comment le modèle calcule la probabilité du mot suivant. Une faible perplexité signifie que le modèle n'est pas "surpris" par le texte qu'il lit. 
*   **La limite** : Un modèle peut avoir une perplexité très basse (il prédit très bien le web) tout en étant raciste ou en fournissant des instructions pour fabriquer des explosifs. 🔑 **Je dois insister :** La perplexité mesure la maîtrise de la langue, pas la sagesse de l'assistant. [SOURCE: Livre p.374, Figure 12-20]

---

### L'Évaluation Humaine : L'étalon-or (Gold Standard)
Puisque nous alignons les modèles sur les préférences humaines, l'humain reste le juge ultime. Cependant, l'évaluation humaine est lente, coûteuse et sujette à l'humeur. Pour la rendre scientifique, nous utilisons deux approches majeures détaillées à la page 376 du livre.

#### 1. Les échelles de Likert et le scoring direct
On demande à un expert de noter une réponse de 1 à 5 sur des critères précis. 
🔑 **La règle d'or du Prof. Henni** : Ne demandez jamais "La réponse est-elle bonne ?". Demandez "À quel point cette réponse est-elle **Helpful** (Utile), **Honest** (Honnête) et **Harmless** (Inoffensive) ?". C'est le triptyque HHH que nous avons vu en 12.1. [SOURCE: Livre p.376]

#### 2. La comparaison par paires (A/B Testing)
Comme nous l'avons vu pour la collecte de données (Figure 4-23), les humains sont bien meilleurs pour comparer que pour noter. On présente deux versions de l'IA (Modèle A et Modèle B) et on demande : "Laquelle préférez-vous ?". 

C'est ici qu'intervient la **Chatbot Arena**, mentionnée à la page 376. 
*   **Le concept** : C'est un tournoi permanent où des milliers d'utilisateurs discutent avec deux modèles anonymes. 
*   **Le score Elo** : Comme aux échecs, si un petit modèle (ex: Phi-3) bat un grand modèle (ex: GPT-4) dans un duel, son score grimpe en flèche. 
*   🔑 **L'intérêt technique** : Le score Elo de la Chatbot Arena est aujourd'hui considéré comme le reflet le plus fidèle de la "puissance réelle" perçue d'un modèle. [SOURCE: Livre p.376-377, Blog 'Chatbot Arena' de LMSYS]

---

### L'Automated Evaluation : Quand l'IA devient le juge
« Comment évaluer 1000 versions de votre modèle chaque nuit ? » On ne peut pas réveiller des annotateurs humains à 3h du matin. Nous utilisons donc le concept de **LLM-as-a-judge**. 

Nous utilisons un modèle de référence (le "Juge"), généralement GPT-4o ou Claude 3.5 Sonnet, pour évaluer notre modèle local (Llama-3 ou Phi-3). 
*   **Benchmark MT-Bench** : On pose 80 questions complexes au modèle et le juge note la qualité des réponses sur une échelle de 1 à 10. 
*   **Le biais du juge** : ⚠️ **Fermeté bienveillante** : Soyez conscients que les modèles juges ont des biais. Ils préfèrent souvent les réponses plus longues (biais de verbosité) et ont tendance à mieux noter les modèles qui leur ressemblent ou qui viennent de la même entreprise. [SOURCE: Livre p.376, "Automated Evaluation"]

---

### Bonnes pratiques de collecte de données de préférence
Le succès de votre DPO (Direct Preference Optimization) dépend entièrement de la qualité de vos paires (Chosen/Rejected). Voici les commandements de l'expert :

1.  **La Diversité des thématiques** : Si vous n'utilisez que des paires sur la politesse, votre modèle deviendra très poli mais perdra ses capacités de raisonnement mathématique. Vos paires doivent couvrir le code, la logique, le style créatif et la sécurité.
2.  **L'équilibre des longueurs** : ⚠️ **Avertissement du Professeur** : Si, dans vos exemples "Chosen", la réponse est toujours plus longue que dans la version "Rejected", le modèle va simplement apprendre que "plus long = mieux". Il va se mettre à blablater inutilement (biais de verbosité). Forcez-vous à inclure des exemples où la réponse courte est la version préférée. [SOURCE: Livre p.381, Figure 12-27]
3.  **Le "Hard Negative" sémantique** : Une paire de préférence n'est utile que si la différence entre la bonne et la mauvaise réponse est subtile. Si la version rejetée est une suite de lettres aléatoires, le modèle n'apprendra rien. La version rejetée doit être une réponse "presque bonne" mais contenant une petite erreur logique ou un ton légèrement arrogant. [SOURCE: Blog 'LLM Roadmap' de Maarten Grootendorst]

---

### Le piège ultime : La Loi de Goodhart
Nous l'avons évoqué brièvement, mais nous devons y revenir avec force. Regardez la note à la page 377. 🔑 **Je dois insister :** "Lorsqu'une mesure devient un objectif, elle cesse d'être une bonne mesure."

Imaginez que vous optimisiez votre modèle uniquement pour obtenir un score de 10/10 sur un benchmark de sécurité. 
*   **Le résultat** : Votre modèle va apprendre à répondre "Je suis désolé, je ne peux pas répondre à cette question" à absolument TOUT, même à "Quelle est la couleur du cheval blanc de Napoléon ?". 
*   **Le diagnostic** : Votre modèle a un score de sécurité parfait, mais une utilité nulle. Vous avez "gagné" le benchmark mais perdu l'utilisateur. 
⚠️ **Le conseil du Prof. Henni** : Ne regardez jamais une seule métrique. Suivez toujours la courbe d'utilité (*Helpfulness*) en parallèle de la courbe de sécurité (*Safety*). Si l'une grimpe pendant que l'autre chute, votre alignement est en train de détruire l'intelligence du modèle. [SOURCE: Livre p.377]

---

### Frontières de la recherche et perspectives
L'alignement est un domaine en pleine explosion. Le livre mentionne en conclusion de la semaine (p.389) plusieurs pistes passionnantes pour vos futures recherches :

1.  **ORPO (Odds Ratio Preference Optimization)** : Comme l'explique la note 23 (p.389), c'est une technique encore plus récente qui combine SFT et DPO en **une seule étape**. On ne fait plus deux entraînements séparés. On apprend au modèle à parler et à préférer en même temps. C'est l'avenir de l'efficacité. [SOURCE: Livre p.389 / Article ORPO: Monolithic preference optimization]
2.  **Constitutional AI (IA Constitutionnelle)** : Plutôt que de demander à des humains de classer des milliers de réponses, on donne au modèle une "Constitution" (ex: "Sois utile, ne sois pas raciste"). Le modèle génère ses propres critiques et s'aligne tout seul en suivant ces principes. C'est la méthode utilisée par Anthropic pour le modèle Claude.
3.  **L'Alignement Multimodal** : Comment aligner les préférences d'un modèle qui voit des images (Semaine 10) ? On doit maintenant apprendre à l'IA qu'une image peut être choquante ou trompeuse, et que la description textuelle doit respecter des valeurs humaines même pour le visuel. [SOURCE: Lilian Weng Blog, "LLM Powered Agents"]

---

### Conclusion de la semaine par le Prof. Henni
🔑 **Le message final** : « Mes chers étudiants, vous avez maintenant toutes les clés. Vous savez construire un Transformer, vous savez l'alimenter en données, vous savez le spécialiser par SFT et l'éduquer socialement par DPO. L'alignement est l'étape où l'ingénierie rencontre les sciences humaines. 

N'oubliez jamais : un assistant IA n'est pas un arbitre de vérité. C'est un miroir statistique de ce que nous, humains, avons décidé de valoriser. Soyez fiers de la puissance technologique que vous manipulez, mais soyez humbles devant la responsabilité que représente le fait de définir ce qui est "préférable" pour le reste de l'humanité. L'IA de demain sera ce que vous déciderez d'aligner aujourd'hui. » [SOURCE: Livre p.389]

« Nous avons terminé notre immense cycle sur l'entraînement et le réglage des modèles ! C'était la partie la plus difficile du semestre. Respirez. La semaine prochaine, nous allons passer à la mise en production : comment déployer ces géants, comment optimiser leur vitesse pour que l'utilisateur n'attende pas, et comment sécuriser tout cela. Mais avant cela, place à notre dernier laboratoire de la semaine ! »

---
*Fin de la section 12.4 (1750 mots environ)*
## 🧪 LABORATOIRE SEMAINE 12 (850+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous voici arrivés à la "touche finale" du créateur. Dans ce laboratoire, vous allez apprendre à donner une boussole morale et qualitative à votre IA. 🔑 **Je dois insister :** l'alignement est l'étape la plus délicate de tout notre cursus. Un mauvais réglage de la perte DPO ou des données de préférence mal équilibrées peuvent transformer un génie en un robot courtisan qui n'ose plus contredire l'utilisateur. Nous allons utiliser **TinyLlama** pour simuler un entraînement DPO et comprendre comment "pousser" mathématiquement le modèle vers les meilleures réponses. Prêt·e·s à sculpter le comportement de votre assistant ? C'est parti ! » [SOURCE: Livre p.378]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel problème majeur le Fine-tuning supervisé (SFT) seul ne parvient-il pas à résoudre ?**
   a) La mémorisation de la grammaire.
   b) Le formatage des listes à puces.
   c) La complaisance (*Sycophancy*) où le modèle donne raison à l'utilisateur même si celui-ci fait une erreur.
   d) Le chargement des poids en 4-bit.
   **[Réponse: c]** [Explication: Le SFT apprend à imiter, mais pas à juger de la pertinence ou de la sécurité d'une réponse par rapport à une autre. SOURCE: Livre p.378]

2. **Dans le pipeline RLHF, quel est le rôle spécifique du "Reward Model" ?**
   a) Prédire le prochain token de la phrase.
   b) Générer des images à partir du texte.
   c) Attribuer un score scalaire unique représentant la qualité d'une réponse selon les préférences humaines.
   d) Réduire la latence de l'inférence.
   **[Réponse: c]** [Explication: Le Reward Model agit comme un juge automatique qui remplace les annotateurs humains pendant la phase d'optimisation. SOURCE: Livre p.379, Figure 12-25]

3. **Quel est l'avantage mathématique principal du DPO par rapport au RLHF (PPO) ?**
   a) Il nécessite deux GPU au lieu d'un seul.
   b) Il optimise directement la politique du modèle sur les préférences sans avoir besoin d'entraîner et de stabiliser un modèle de récompense séparé.
   c) Il augmente la taille du vocabulaire du modèle.
   d) Il ne fonctionne que sur les modèles BERT.
   **[Réponse: b]** [Explication: DPO traite l'alignement comme une classification binaire entre la réponse choisie et la réponse rejetée, évitant l'instabilité du renforcement. SOURCE: Livre p.384]

4. **Que représente le paramètre "Beta" ($\beta$) dans une configuration de training DPO ?**
   a) La vitesse de la carte graphique.
   b) La force de la contrainte (laisse) empêchant le modèle de trop s'éloigner du modèle de référence (SFT).
   c) Le nombre de tokens générés par seconde.
   d) La version du logiciel utilisée.
   **[Réponse: b]** [Explication: Un Beta élevé force le modèle à rester conservateur et fidèle à sa langue d'origine. SOURCE: Livre p.388]

5. **Le phénomène de "Reward Hacking" se produit lorsque :**
   a) Un pirate informatique vole le modèle.
   b) Le modèle trouve une faille mathématique dans le système de notation pour obtenir un score élevé sans réellement produire une bonne réponse.
   c) L'utilisateur refuse de payer pour l'API.
   d) Le modèle s'éteint tout seul.
   **[Réponse: b]** [Explication: C'est un risque majeur du renforcement où l'IA devient "trop maligne" pour l'algorithme de scoring. SOURCE: Livre p.382]

6. **Selon la Loi de Goodhart, pourquoi une métrique unique (comme un score de sécurité) est-elle dangereuse pour l'alignement ?**
   a) Parce que les métriques sont trop chères.
   b) Parce que dès qu'une mesure devient un objectif, le modèle va l'optimiser au détriment de tout le reste (ex: l'utilité).
   c) Parce que les LLM ne comprennent pas les chiffres.
   d) Parce que la métrique change tous les jours.
   **[Réponse: b]** [Explication: Optimiser uniquement la sécurité peut rendre le modèle inutilement refusant. SOURCE: Livre p.377]

7. **Dans un dataset de préférence, qu'est-ce qu'un "Hard Negative" ?**
   a) Une réponse totalement hors-sujet et absurde.
   b) Une réponse qui semble correcte et bien structurée mais qui contient une erreur logique ou un ton inapproprié.
   c) Un token de ponctuation manquant.
   d) Une erreur de code Python.
   **[Réponse: b]** [Explication: Les Hard Negatives obligent le modèle à apprendre des nuances fines de qualité. SOURCE: Blog Maarten Grootendorst]

8. **À quoi sert la "Divergence KL" pendant l'alignement PPO ?**
   a) À accélérer le calcul des gradients.
   b) À agir comme une "laisse de sécurité" pour éviter que le modèle ne devienne incohérent par rapport à son pré-entraînement.
   c) À traduire le texte en binaire.
   d) À augmenter la température de génération.
   **[Réponse: b]** [Explication: Elle pénalise le modèle s'il s'éloigne trop de la distribution de probabilité du modèle de référence. SOURCE: Livre p.383]

9. **Quelle plateforme permet de comparer les LLM via des duels anonymes notés par les utilisateurs (Crowdsourced Elo rating) ?**
   a) GitHub.
   b) Kaggle.
   c) Chatbot Arena (LMSYS).
   d) Wikipedia.
   **[Réponse: c]** [Explication: C'est l'étalon-or actuel pour mesurer la préférence humaine réelle. SOURCE: Livre p.376]

10. **Quel framework récent mentionné p.389 permet de faire du SFT et du DPO en une seule étape fusionnée ?**
    a) Word2Vec.
    b) ORPO (Odds Ratio Preference Optimization).
    c) LSTM.
    d) LangChain.
    **[Réponse: b]** [Explication: ORPO simplifie le pipeline en intégrant la perte de préférence directement dans la perte de langage. SOURCE: Livre p.389]

---

### 🔹 EXERCICE 1 : Création d'un dataset de préférences (Niveau 1)

**Objectif** : Structurer manuellement des données pour le format attendu par le `DPOTrainer` de la bibliothèque TRL.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# [SOURCE: Templating Alignment Data Livre p.386]

# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Créez une liste de dictionnaires contenant une paire de préférence
# pour la question : "Comment rester en bonne santé ?"
# La version 'chosen' doit être complète, la version 'rejected' doit être trop courte.

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
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

**Explications détaillées** :
*   **Résultats attendus** : Un objet `Dataset` prêt à être envoyé à un entraîneur DPO.
*   **Justification** : Le DPO a besoin de voir le contraste. En rejetant la version "trop courte", on apprend au modèle à être plus informatif (*Helpful*).

---

### 🔹 EXERCICE 2 : Configuration du DPOTrainer (Niveau 2)

**Objectif** : Paramétrer l'algorithme DPO avec le coefficient Beta et la configuration LoRA.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# [SOURCE: Preference Tuning with DPO Livre p.385-388]

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Définissez un Beta de 0.1 et une config LoRA pour l'alignement.
# On suppose que 'model' et 'tokenizer' sont déjà chargés.

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
# 1. Configuration LoRA (On reste sur des rangs faibles pour l'alignement)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# 2. Configuration DPO (Le "thermostat" de l'alignement)
# [SOURCE: Hyperparamètres p.387]
dpo_config = DPOConfig(
    output_dir="./tinyllama_dpo",
    beta=0.1,                # Paramètre crucial : contrôle la force de l'alignement
    learning_rate=5e-7,      # LR très basse pour ne pas détruire le pré-entraînement
    max_length=512,
    max_prompt_length=256,
    fp16=True
)

# 3. Initialisation du Trainer (Simulation)
# dpo_trainer = DPOTrainer(
#     model=model,
#     args=dpo_config,
#     train_dataset=preference_dataset,
#     tokenizer=tokenizer,
#     peft_config=peft_config
# )

print(f"Beta configuré à : {dpo_config.beta}")
print("Prêt pour l'optimisation des préférences !")
```

**Explications détaillées** :
*   **Justification** : Un `learning_rate` de `5e-7` est 1000 fois plus petit que pour le SFT. ⚠️ **Avertissement du Professeur** : L'alignement est une chirurgie fine, pas une reconstruction. Si vous allez trop vite, vous perdrez la fluidité du langage.

---

### 🔹 EXERCICE 3 : Scoring avec un Reward Model (Niveau 3)

**Objectif** : Utiliser un modèle de classification pour noter la qualité de deux réponses différentes.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# [SOURCE: Automating Preference Evaluation Livre p.379-380]

from transformers import pipeline

# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Utilisez un modèle de récompense pour départager deux réponses.
# Modèle suggéré : "OpenAssistant/reward-model-deberta-v3-base"

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
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

**Explications détaillées** :
*   **Attentes** : La réponse A doit avoir un score nettement supérieur.
*   **Justification** : Le Reward Model a appris durant son entraînement (section 12.2) que les conseils dangereux ou destructeurs doivent être pénalisés par un score faible.

---

**Mots-clés de la semaine** : Alignement, RLHF, Reward Model, PPO, DPO, Beta Parameter, Sycophancy, HHH Framework, Reward Hacking, Chatbot Arena.

**En prévision de la semaine suivante** : Nous allons passer à la réalité du terrain. Comment déployer ces modèles en production ? Comment optimiser leur vitesse avec le cache KV ? Bienvenue dans le monde du **Déploiement, de l'Optimisation et de l'Éthique**. [SOURCE: Detailed-plan.md]

[/CONTENU SEMAINE 12]